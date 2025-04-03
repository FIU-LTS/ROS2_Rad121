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

class RadiationHeatmapNode : public rclcpp::Node
{
public:
    RadiationHeatmapNode()
        : Node("radiation_heatmap_node"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        std::stringstream ss;
        std::time_t now = std::time(nullptr);
        ss << "rad_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
        session_folder_ = ss.str();
        std::filesystem::create_directory(session_folder_);

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&RadiationHeatmapNode::mapCallback, this, std::placeholders::_1));
        rad_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/rad", 10, std::bind(&RadiationHeatmapNode::radCallback, this, std::placeholders::_1));
        rad_costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rad_heatmap", 10);

        RCLCPP_INFO(this->get_logger(), "Radiation Heatmap Node Initialized. Session folder: %s", session_folder_.c_str());
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

    bool isVisible(int x0, int y0, int x1, int y1)
    {
        int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
        int width = base_map_.info.width;

        while (true)
        {
            int idx = y0 * width + x0;
            if (base_map_.data[idx] >= 50)
                return false;
            if (x0 == x1 && y0 == y1)
                break;
            int e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
        return true;
    }

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        bool map_changed = base_map_.info.width != msg->info.width ||
                           base_map_.info.height != msg->info.height ||
                           base_map_.info.resolution != msg->info.resolution ||
                           base_map_.info.origin.position.x != msg->info.origin.position.x ||
                           base_map_.info.origin.position.y != msg->info.origin.position.y;

        base_map_ = *msg;

        if (rad_costmap_.data.empty() || map_changed)
        {
            rad_costmap_ = *msg;
            rad_costmap_.data.assign(msg->info.width * msg->info.height, 0);
            rad_field_ = cv::Mat::zeros(msg->info.height, msg->info.width, CV_32FC1);
            RCLCPP_INFO(this->get_logger(), "Rad costmap and field initialized/resized to %ux%u", msg->info.width, msg->info.height);
        }
    }

    void radCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data[0] < 400.0)
            return;

        geometry_msgs::msg::PoseStamped pose_in, pose_out;
        pose_in.header.frame_id = "base_link";
        pose_in.header.stamp = this->now();

        try
        {
            tf2::doTransform(pose_in, pose_out, tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero));
        }
        catch (tf2::TransformException &)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(map_mutex_);
        samples_.push_back({pose_out.pose.position.x, pose_out.pose.position.y, msg->data[0]});
        updateRadiationField();
    }

    void updateRadiationField()
    {
        rad_field_ = cv::Mat::zeros(base_map_.info.height, base_map_.info.width, CV_32FC1);
        double res = base_map_.info.resolution, sigma = 5.0, inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
        double origin_x = base_map_.info.origin.position.x, origin_y = base_map_.info.origin.position.y;

        for (int y = 0; y < rad_field_.rows; ++y)
        {
            for (int x = 0; x < rad_field_.cols; ++x)
            {
                double wx = origin_x + x * res + res / 2.0, wy = origin_y + y * res + res / 2.0;
                double numerator = 0.0, denominator = 0.0;

                for (const auto &sample : samples_)
                {
                    if (sample.cpm < 100.0)
                        continue;

                    int sx = (sample.x - origin_x) / res, sy = (sample.y - origin_y) / res;
                    if (sx < 0 || sx >= rad_field_.cols || sy < 0 || sy >= rad_field_.rows)
                        continue;
                    if (!isVisible(sx, sy, x, y))
                        continue;

                    double dx = wx - sample.x, dy = wy - sample.y, dist2 = dx * dx + dy * dy;
                    if (dist2 > 4.0)
                        continue;

                    double weight = std::exp(-dist2 * inv_2sigma2);
                    numerator += weight * sample.cpm;
                    denominator += weight;
                }

                rad_field_.at<float>(y, x) = (denominator > 0.0) ? static_cast<float>(numerator / denominator) : 0.0f;
            }
        }

        cv::GaussianBlur(rad_field_, rad_field_, cv::Size(15, 15), 5.0);

        const float background_threshold = 50.0f;
        std::vector<float> values;
        for (int y = 0; y < rad_field_.rows; ++y)
        {
            for (int x = 0; x < rad_field_.cols; ++x)
            {
                float &val = rad_field_.at<float>(y, x);
                if (val < background_threshold)
                    val = 0.0f;
                else
                    values.push_back(val);
            }
        }

        min_cpm_ = 400.0f, max_cpm_ = 10000.0f;
        if (values.size() >= 50)
        {
            std::sort(values.begin(), values.end());
            min_cpm_ = std::max(400.0f, values[values.size() * 0.05]);
            max_cpm_ = values[values.size() * 0.95];
            if (max_cpm_ - min_cpm_ < 1.0f)
                max_cpm_ = min_cpm_ + 1.0f;
        }

        float log_min = std::log(min_cpm_), log_max = std::log(max_cpm_);

        for (int y = 0; y < rad_field_.rows; ++y)
        {
            for (int x = 0; x < rad_field_.cols; ++x)
            {
                float val = rad_field_.at<float>(y, x);
                if (val < background_threshold)
                {
                    rad_costmap_.data[y * rad_field_.cols + x] = -1;
                }
                else
                {
                    float log_val = std::log(std::max(val, min_cpm_));
                    float scaled = 100.0f * (log_val - log_min) / (log_max - log_min);
                    int8_t bucketed = static_cast<int8_t>(std::round(scaled / 5.0f) * 5.0f);
                    rad_costmap_.data[y * rad_field_.cols + x] = bucketed;
                }
            }
        }

        cv::Mat norm_field, color_field;
        cv::Mat scaled_field = cv::Mat::zeros(rad_field_.size(), CV_8UC1);
        for (int y = 0; y < rad_field_.rows; ++y)
        {
            for (int x = 0; x < rad_field_.cols; ++x)
            {
                float val = rad_field_.at<float>(y, x);
                if (val < background_threshold)
                {
                    scaled_field.at<uchar>(y, x) = 0;
                }
                else
                {
                    float log_val = std::log(std::max(val, min_cpm_));
                    float scaled = 255.0f * (log_val - log_min) / (log_max - log_min);
                    scaled_field.at<uchar>(y, x) = static_cast<uchar>(std::clamp(scaled, 0.0f, 255.0f));
                }
            }
        }
        cv::applyColorMap(scaled_field, color_field, cv::COLORMAP_JET);
        // Create the grayscale base map image
        cv::Mat base_gray(rad_field_.rows, rad_field_.cols, CV_8UC1);
        for (int y = 0; y < rad_field_.rows; ++y)
        {
            for (int x = 0; x < rad_field_.cols; ++x)
            {
                int idx = y * rad_field_.cols + x;
                int8_t occ = base_map_.data[idx];
                base_gray.at<uchar>(y, x) = (occ == -1) ? 127 : (occ == 100 ? 0 : 255);
            }
        }

        // Convert grayscale to BGR
        cv::Mat base_bgr;
        cv::cvtColor(base_gray, base_bgr, cv::COLOR_GRAY2BGR);

        // Blend with the heatmap
        cv::Mat blended;
        cv::addWeighted(base_bgr, 0.5, color_field, 0.5, 0.0, blended);
        int legend_bar_height = 40;
        int legend_padding = 10;
        int tick_font = cv::FONT_HERSHEY_SIMPLEX;
        float tick_font_scale = 0.4;
        int tick_thickness = 1;
        
        int bar_width = blended.cols - 2 * legend_padding;
        cv::Mat legend_bar(legend_bar_height + 30, blended.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        
        // Build horizontal color gradient
        for (int x = 0; x < bar_width; ++x)
        {
            uchar value = static_cast<uchar>((float)x / bar_width * 255);
            cv::Mat input_pixel(1, 1, CV_8UC1, cv::Scalar(value));
            cv::Mat color_pixel;
            cv::applyColorMap(input_pixel, color_pixel, cv::COLORMAP_JET);
            cv::Vec3b color = color_pixel.at<cv::Vec3b>(0, 0);
                        for (int y = legend_padding; y < legend_bar_height; ++y)
            {
                legend_bar.at<cv::Vec3b>(y, x + legend_padding) = color;
            }
        }
        
        // Draw min and max text
        char min_buf[16], max_buf[16];
        snprintf(min_buf, sizeof(min_buf), "%.0f", min_cpm_);
        snprintf(max_buf, sizeof(max_buf), "%.0f", max_cpm_);
        cv::putText(legend_bar, min_buf, cv::Point(legend_padding, legend_bar_height + 25), tick_font, tick_font_scale, cv::Scalar(0, 0, 0), tick_thickness);
        cv::putText(legend_bar, max_buf, cv::Point(legend_padding + bar_width - 40, legend_bar_height + 25), tick_font, tick_font_scale, cv::Scalar(0, 0, 0), tick_thickness);
        
        // Label
        std::string label = "Radiation Level (CPM)";
        int base_line = 0;
        cv::Size text_size = cv::getTextSize(label, tick_font, 0.5, 1, &base_line);
        cv::putText(legend_bar, label,
                    cv::Point((legend_bar.cols - text_size.width) / 2, legend_padding - 2 + text_size.height),
                    tick_font, 0.5, cv::Scalar(0, 0, 0), 1);
        
        // ---- Combine with heatmap ----
        cv::Mat full_img(blended.rows + legend_bar.rows, blended.cols, CV_8UC3);
        blended.copyTo(full_img(cv::Rect(0, 0, blended.cols, blended.rows)));
        legend_bar.copyTo(full_img(cv::Rect(0, blended.rows, legend_bar.cols, legend_bar.rows)));
        
        latest_blended_image_ = full_img.clone(); // Save for export
        // Flip if needed (to match map orientation)
        cv::flip(blended, blended, 0);

        std_msgs::msg::Header header;
        header.frame_id = "map";

        sensor_msgs::msg::Image::SharedPtr msg =
            cv_bridge::CvImage(header, "bgr8", blended).toImageMsg();

        if (rad_image_pub_.getNumSubscribers() > 0)
        {
            rad_image_pub_.publish(msg);
        }

        rad_costmap_.header.stamp = this->now();
        rad_costmap_pub_->publish(rad_costmap_);
    }

    void saveMap()
    {
        std::string output_path = session_folder_ + "/map_with_costmap.png";
        std::lock_guard<std::mutex> lock(map_mutex_);
        
        if (latest_blended_image_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No blended image available to save.");
            return;
        }
    
        cv::imwrite(output_path, latest_blended_image_);
        cv::imwrite(session_folder_ + "/raw_field.png", rad_field_);
        RCLCPP_INFO(this->get_logger(), "Saved blended heatmap to: %s", output_path.c_str());
    }

    std::string session_folder_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rad_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr rad_costmap_pub_;
    image_transport::Publisher rad_image_pub_;
    cv::Mat latest_blended_image_;


    nav_msgs::msg::OccupancyGrid base_map_, rad_costmap_;
    cv::Mat rad_field_;
    std::vector<RadiationSample> samples_;
    std::mutex map_mutex_;
    float min_cpm_ = 400.0f;
    float max_cpm_ = 10000.0f;
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