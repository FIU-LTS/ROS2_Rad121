#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <vector>
#include <mutex>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>

class RadiationHeatmapNode : public rclcpp::Node
{
public:
    RadiationHeatmapNode()
        : Node("radiation_heatmap_node"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        this->declare_parameter<std::string>("sensor_frame", "base_link");
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<double>("min_valid_cpm", 50.0);
        this->declare_parameter<double>("min_sample_cpm", 100.0);
        this->declare_parameter<double>("gaussian_sigma", 5.0);
        this->declare_parameter<int>("gaussian_blur_size", 15); // Should be odd
        this->declare_parameter<double>("distance_cutoff", 10.0);
        this->declare_parameter<double>("background_threshold", 400.0);

        sensor_frame_ = this->get_parameter("sensor_frame").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        min_valid_cpm_ = this->get_parameter("min_valid_cpm").as_double();
        min_sample_cpm_ = this->get_parameter("min_sample_cpm").as_double();
        gaussian_sigma_ = this->get_parameter("gaussian_sigma").as_double();
        gaussian_blur_size_ = this->get_parameter("gaussian_blur_size").as_int();
        distance_cutoff_ = this->get_parameter("distance_cutoff").as_double();
        background_threshold_ = this->get_parameter("background_threshold").as_double();

        // Initialize overall max CPM value
        overall_max_cpm_ = background_threshold_;

        std::stringstream ss;
        std::time_t now = std::time(nullptr);
        ss << "rad_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
        session_folder_ = ss.str();
        std::filesystem::create_directory(session_folder_);

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(), // Latching QoS
            std::bind(&RadiationHeatmapNode::mapCallback, this, std::placeholders::_1));
        rad_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/rad", 10, std::bind(&RadiationHeatmapNode::radCallback, this, std::placeholders::_1));
        rad_costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rad_heatmap", 10);

        RCLCPP_INFO(this->get_logger(), "Radiation Heatmap Node Initialized. Session folder: %s", session_folder_.c_str());
        RCLCPP_INFO(this->get_logger(), "Waiting for map...");

        rclcpp::on_shutdown([this]()
                            { saveMap(); });
    }

    void init_image_transport()
    {
        image_transport::ImageTransport it(shared_from_this());
        rad_image_pub_ = it.advertise("/rad_heatmap_image", 1);
    }

private:
    struct RadiationSample
    {
        double x, y, cpm;
    };

    double overall_max_cpm_; // Stores the highest CPM value encountered

    bool isVisible(int x0, int y0, int x1, int y1)
    {
        if (base_map_.info.width == 0 || base_map_.info.height == 0) return false;

        int width = base_map_.info.width;
        int height = base_map_.info.height;

        auto in_bounds = [&](int x, int y) {
            return x >= 0 && x < width && y >= 0 && y < height;
        };

        int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;

        while (true)
        {
            if (!in_bounds(x0, y0)) return false;

            int idx = y0 * width + x0;
            if (idx < 0 || static_cast<size_t>(idx) >= base_map_.data.size()) {
                 RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "isVisible check: Index out of bounds.");
                 return false;
            }
            if (base_map_.data[idx] >= 50) return false; // Occupied cell threshold
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
        return true;
    }

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        if (msg->info.width == 0 || msg->info.height == 0) {
            RCLCPP_WARN(this->get_logger(), "Received map with zero dimensions (width=%u, height=%u). Ignoring.", msg->info.width, msg->info.height);
            return;
        }

        const size_t expected_size = msg->info.width * msg->info.height;
        bool need_reinit = false;

        if (rad_costmap_.info.width != msg->info.width ||
            rad_costmap_.info.height != msg->info.height ||
            rad_field_.empty() ||
            static_cast<unsigned int>(rad_field_.rows) != msg->info.height ||
            static_cast<unsigned int>(rad_field_.cols) != msg->info.width)
        {
             need_reinit = true;
             RCLCPP_INFO(this->get_logger(), "Received map. Initializing/Resizing structures for size %u x %u", msg->info.width, msg->info.height);
        }

        base_map_ = *msg; // Update base map data and metadata

        if (need_reinit)
        {
            rad_costmap_.header = msg->header;
            rad_costmap_.info = msg->info;
            rad_costmap_.data.assign(expected_size, -1); // Initialize costmap data

            rad_field_ = cv::Mat::zeros(msg->info.height, msg->info.width, CV_32FC1);
        }
    }

    void radCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.empty() || msg->data[0] < min_valid_cpm_) return;

        geometry_msgs::msg::PoseStamped pose_in, pose_out;
        pose_in.header.frame_id = sensor_frame_;
        pose_in.header.stamp = this->now();

        try {
            pose_out = tf_buffer_.transform(pose_in, map_frame_, tf2::durationFromSec(0.1));
        } catch (const tf2::TransformException &ex) {
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "Could not transform %s to %s: %s",
                                  sensor_frame_.c_str(), map_frame_.c_str(), ex.what());
            return;
        }

        double current_cpm = msg->data[0];

        std::lock_guard<std::mutex> lock(map_mutex_);

         if (base_map_.info.width == 0 || base_map_.info.height == 0 || rad_field_.empty()) {
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Map not ready, skipping radiation sample processing.");
             return;
         }

        if (current_cpm > overall_max_cpm_) {
            overall_max_cpm_ = current_cpm;
            RCLCPP_INFO(this->get_logger(), "New overall maximum radiation recorded: %.2f CPM", overall_max_cpm_);
        }

        samples_.push_back({pose_out.pose.position.x, pose_out.pose.position.y, current_cpm});
        updateRadiationField();
    }

    void updateRadiationField()
    {
        if (base_map_.info.width == 0 || base_map_.info.height == 0 || rad_field_.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "updateRadiationField called before map/rad_field initialized. Skipping.");
            return;
        }

        cv::Mat current_rad_field = cv::Mat::zeros(base_map_.info.height, base_map_.info.width, CV_32FC1);
        double res = base_map_.info.resolution;
        double origin_x = base_map_.info.origin.position.x;
        double origin_y = base_map_.info.origin.position.y;
        double inv_2sigma2 = 1.0 / (2.0 * gaussian_sigma_ * gaussian_sigma_);
        double distance_cutoff_sq = distance_cutoff_ * distance_cutoff_;

        for (int y = 0; y < current_rad_field.rows; ++y) {
            for (int x = 0; x < current_rad_field.cols; ++x) {
                double wx = origin_x + x * res + res / 2.0;
                double wy = origin_y + y * res + res / 2.0;
                double numerator = 0.0, denominator = 0.0;

                for (const auto &sample : samples_) {
                    if (sample.cpm < min_sample_cpm_) continue;

                    int sx = static_cast<int>((sample.x - origin_x) / res);
                    int sy = static_cast<int>((sample.y - origin_y) / res);

                    if (sx < 0 || sx >= current_rad_field.cols || sy < 0 || sy >= current_rad_field.rows) continue;
                    if (!isVisible(sx, sy, x, y)) continue;

                    double dx = wx - sample.x, dy = wy - sample.y;
                    double dist2 = dx * dx + dy * dy;
                    if (dist2 > distance_cutoff_sq) continue;

                    double weight = std::exp(-dist2 * inv_2sigma2);
                    numerator += weight * sample.cpm;
                    denominator += weight;
                }
                current_rad_field.at<float>(y, x) = (denominator > 1e-9) ? static_cast<float>(numerator / denominator) : 0.0f;
            }
        }
         rad_field_ = current_rad_field;

        int blur_ksize = gaussian_blur_size_;
        if (blur_ksize % 2 == 0) {
            blur_ksize++;
            RCLCPP_WARN_ONCE(this->get_logger(), "Gaussian blur size (%d) was even, adjusting to %d.", gaussian_blur_size_, blur_ksize);
        }
        if (blur_ksize <= 0) {
             blur_ksize = 1;
             RCLCPP_WARN_ONCE(this->get_logger(), "Gaussian blur size (%d) was non-positive, adjusting to %d.", gaussian_blur_size_, blur_ksize);
        }
        cv::GaussianBlur(rad_field_, rad_field_, cv::Size(blur_ksize, blur_ksize), gaussian_sigma_);

        float true_min_cpm = std::numeric_limits<float>::max();
        float true_max_cpm = std::numeric_limits<float>::lowest();
        bool found_value_above_threshold = false;

        for (int y = 0; y < rad_field_.rows; ++y) {
            for (int x = 0; x < rad_field_.cols; ++x) {
                float &val = rad_field_.at<float>(y, x);
                if (val < background_threshold_) {
                    val = 0.0f;
                } else {
                    true_min_cpm = std::min(true_min_cpm, val);
                    true_max_cpm = std::max(true_max_cpm, val);
                    found_value_above_threshold = true;
                }
            }
        }

        if (!found_value_above_threshold) {
            true_min_cpm = background_threshold_;
            true_max_cpm = background_threshold_ + 1.0f;
             if (!samples_.empty()) {
                 RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,"No radiation values found above background threshold: %.2f", background_threshold_);
             }
        } else if (true_max_cpm <= true_min_cpm) {
            true_max_cpm = true_min_cpm + 1.0f;
        }

        float log_min = std::log(std::max(true_min_cpm, 1.0f));
        float log_max = std::log(std::max(true_max_cpm, 1.0f));
        float log_range = log_max - log_min;

        if (log_range <= 1e-6) {
            log_range = 1.0f;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,"Log range too small (Min: %.2f, Max: %.2f). Using default range for scaling.", true_min_cpm, true_max_cpm);
            float center_log = std::log(std::max(true_min_cpm, 1.0f));
            log_min = center_log - 0.5f;
            log_max = center_log + 0.5f;
            log_range = 1.0f;
        }

         for (int y = 0; y < rad_field_.rows; ++y) {
             for (int x = 0; x < rad_field_.cols; ++x) {
                 int idx = y * rad_field_.cols + x;
                 if (idx < 0 || static_cast<size_t>(idx) >= rad_costmap_.data.size()) {
                     RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Costmap index out of bounds (%d).", idx);
                     continue;
                 }

                 float val = rad_field_.at<float>(y, x);
                 if (val < background_threshold_) {
                     rad_costmap_.data[idx] = -1;
                 } else {
                     float log_val = std::log(std::max(val, 1.0f));
                     float scaled = 100.0f * (log_val - log_min) / log_range;
                     int8_t bucketed = static_cast<int8_t>(std::round(std::clamp(scaled, 0.0f, 100.0f) / 5.0f) * 5.0f);
                     rad_costmap_.data[idx] = std::max(static_cast<int8_t>(0), std::min(bucketed, static_cast<int8_t>(100)));
                 }
             }
         }

        cv::Mat base_gray(rad_field_.rows, rad_field_.cols, CV_8UC1);
        for (int y = 0; y < rad_field_.rows; ++y) {
            for (int x = 0; x < rad_field_.cols; ++x) {
                int idx = y * rad_field_.cols + x;
                 if (idx < 0 || static_cast<size_t>(idx) >= base_map_.data.size()) {
                     RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Base map index out of bounds (%d).", idx);
                     base_gray.at<uchar>(y, x) = 127;
                     continue;
                 }
                int8_t occ = base_map_.data[idx];
                base_gray.at<uchar>(y, x) = (occ == -1) ? 127 : (occ >= 50 ? 0 : 255);
            }
        }
        cv::Mat base_bgr;
        cv::cvtColor(base_gray, base_bgr, cv::COLOR_GRAY2BGR);

        cv::Mat scaled_field = cv::Mat::zeros(rad_field_.size(), CV_8UC1);
        for (int y = 0; y < rad_field_.rows; ++y) {
            for (int x = 0; x < rad_field_.cols; ++x) {
                float val = rad_field_.at<float>(y, x);
                if (val >= background_threshold_) {
                    float log_val = std::log(std::max(val, 1.0f));
                    float scaled = 255.0f * (log_val - log_min) / log_range;
                    scaled_field.at<uchar>(y, x) = static_cast<uchar>(std::clamp(scaled, 0.0f, 255.0f));
                }
            }
        }
        cv::Mat color_field;
        cv::applyColorMap(scaled_field, color_field, cv::COLORMAP_JET);

        cv::Mat mask;
        cv::compare(rad_field_, background_threshold_, mask, cv::CMP_LT);
        color_field.setTo(cv::Scalar(0, 0, 0), mask); // Set areas below threshold to black

        cv::Mat blended;
         if (base_bgr.size() != color_field.size()) {
             RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Size mismatch between base map and color field before blending.");
             return;
         }
        cv::addWeighted(base_bgr, 0.5, color_field, 0.5, 0.0, blended);

        int legend_width = 50;
         if (blended.empty()) {
             RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Blended image is empty before adding legend.");
             return;
         }
        int legend_height = blended.rows / 2;
        int legend_x = blended.cols;
        int legend_y = blended.rows / 4;
        int border_width = legend_width + 60;

        cv::copyMakeBorder(blended, blended, 0, 0, 0, border_width, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

        cv::Mat legend(legend_height, legend_width, CV_8UC3);
        for (int i = 0; i < legend_height; ++i) {
            uchar val_map = 255 - static_cast<uchar>((static_cast<float>(i) / legend_height) * 255.0f);
            cv::Mat color_val(1, 1, CV_8UC1, cv::Scalar(val_map));
            cv::Mat mapped_color;
            cv::applyColorMap(color_val, mapped_color, cv::COLORMAP_JET);
             if (!mapped_color.empty() && mapped_color.type() == CV_8UC3) {
                 legend.row(i) = mapped_color.at<cv::Vec3b>(0,0);
             }
        }
         if (legend_x + 10 + legend_width <= blended.cols && legend_y + legend_height <= blended.rows) {
             legend.copyTo(blended(cv::Rect(legend_x + 10, legend_y, legend_width, legend_height)));
         } else {
              RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Legend ROI calculation is outside bordered image bounds.");
         }

         int text_x = legend_x + 15 + legend_width;
         if (text_x < blended.cols) {
             cv::putText(blended,
                     cv::format("%.1f", overall_max_cpm_), // Use overall max for top label
                     cv::Point(text_x, legend_y + 15),
                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

             cv::putText(blended,
                     cv::format("%.1f", true_min_cpm), // Use current min for bottom label
                     cv::Point(text_x, legend_y + legend_height - 5),
                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

             cv::putText(blended,
                     "CPM",
                     cv::Point(text_x, legend_y + legend_height / 2),
                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
         }

        rad_costmap_.header.stamp = this->now();
        rad_costmap_.header.frame_id = map_frame_;
        rad_costmap_pub_->publish(rad_costmap_);

        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = map_frame_;

         if (!blended.empty()) {
             try {
                 sensor_msgs::msg::Image::SharedPtr img_msg =
                     cv_bridge::CvImage(header, "bgr8", blended).toImageMsg();
                 if (rad_image_pub_.getNumSubscribers() > 0) {
                     rad_image_pub_.publish(*img_msg);
                 }
                 latest_blended_image_ = blended.clone(); // Store image with legend
             } catch (const cv_bridge::Exception& e) {
                  RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,"cv_bridge exception: %s", e.what());
             } catch (const cv::Exception& e) {
                  RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,"OpenCV exception during image message creation: %s", e.what());
             }
         } else {
              RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Blended image was empty, skipping image publishing.");
         }
    }

    void saveMap()
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        std::string output_path = session_folder_ + "/map_with_costmap.png";

        if (!latest_blended_image_.empty()) {
            try {
                 bool success_map = cv::imwrite(output_path, latest_blended_image_);
                 if (!success_map) {
                      RCLCPP_ERROR(this->get_logger(), "Failed to save heatmap image to: %s", output_path.c_str());
                 } else {
                      RCLCPP_INFO(this->get_logger(), "Saved heatmap image to: %s", output_path.c_str());
                 }
                 if (!rad_field_.empty()) {
                     // Optionally save the raw field in a compatible format (e.g., scaled grayscale or raw float data)
                     // Saving raw float data requires different handling (e.g., YML/XML storage or a custom format)
                     // Saving as scaled PNG might lose precision:
                     // cv::Mat raw_field_display;
                     // cv::normalize(rad_field_, raw_field_display, 0, 255, cv::NORM_MINMAX, CV_8U);
                     // cv::imwrite(session_folder_ + "/raw_field_scaled.png", raw_field_display);
                 }
            } catch (const cv::Exception& e) {
                 RCLCPP_ERROR(this->get_logger(), "OpenCV Exception during saveMap: %s", e.what());
            }
        } else {
             RCLCPP_WARN(this->get_logger(), "No valid heatmap image generated to save.");
        }
    }

    std::string session_folder_;
    std::string sensor_frame_, map_frame_;
    double min_valid_cpm_, min_sample_cpm_;
    double gaussian_sigma_, distance_cutoff_, background_threshold_;
    int gaussian_blur_size_;

    std::vector<RadiationSample> samples_;
    nav_msgs::msg::OccupancyGrid base_map_, rad_costmap_;
    cv::Mat rad_field_, latest_blended_image_;
    std::mutex map_mutex_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rad_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr rad_costmap_pub_;
    image_transport::Publisher rad_image_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RadiationHeatmapNode>();
    node->init_image_transport();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}