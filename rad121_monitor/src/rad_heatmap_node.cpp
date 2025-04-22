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
#include <filesystem> // Requires C++17
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility> // For std::pair

class RadiationHeatmapNode : public rclcpp::Node
{
private:
    struct RadiationSample
    {
        double x, y, cpm;
    };

    // --- Member Variables ---
    std::string session_folder_;
    std::string sensor_frame_, map_frame_;
    std::string output_directory_;
    double min_valid_cpm_, min_sample_cpm_;
    double gaussian_sigma_, distance_cutoff_, background_threshold_;
    int gaussian_blur_size_;
    double overall_max_cpm_;

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

    // --- Initialization ---
    void declareAndGetParameters() {
        this->declare_parameter<std::string>("sensor_frame", "base_link");
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<std::string>("output_directory", ""); // New parameter for output dir
        this->declare_parameter<double>("min_valid_cpm", 50.0);
        this->declare_parameter<double>("min_sample_cpm", 100.0);
        this->declare_parameter<double>("gaussian_sigma", 5.0);
        this->declare_parameter<int>("gaussian_blur_size", 15);
        this->declare_parameter<double>("distance_cutoff", 10.0);
        this->declare_parameter<double>("background_threshold", 400.0);

        sensor_frame_ = this->get_parameter("sensor_frame").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        output_directory_ = this->get_parameter("output_directory").as_string();
        min_valid_cpm_ = this->get_parameter("min_valid_cpm").as_double();
        min_sample_cpm_ = this->get_parameter("min_sample_cpm").as_double();
        gaussian_sigma_ = this->get_parameter("gaussian_sigma").as_double();
        gaussian_blur_size_ = this->get_parameter("gaussian_blur_size").as_int();
        distance_cutoff_ = this->get_parameter("distance_cutoff").as_double();
        background_threshold_ = this->get_parameter("background_threshold").as_double();

        // Ensure blur size is odd and positive
        if (gaussian_blur_size_ <= 0) gaussian_blur_size_ = 1;
        else if (gaussian_blur_size_ % 2 == 0) gaussian_blur_size_++;
    }

    void setupCommunications() {
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(),
            std::bind(&RadiationHeatmapNode::mapCallback, this, std::placeholders::_1));
        rad_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/rad", 10, std::bind(&RadiationHeatmapNode::radCallback, this, std::placeholders::_1));
        rad_costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rad_heatmap", 10);
    }

    void createSessionFolder() {
        std::stringstream ss;
        std::time_t now = std::time(nullptr);
        ss << "rad_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
        std::string session_name = ss.str();

        std::filesystem::path base_path;
        if (output_directory_.empty()) {
            base_path = std::filesystem::current_path();
            RCLCPP_INFO(this->get_logger(), "Output directory parameter not set. Using current working directory: %s", base_path.c_str());
        } else {
            base_path = output_directory_;
        }

        session_folder_ = (base_path / session_name).string();

        try {
            if (std::filesystem::create_directories(session_folder_)) {
                RCLCPP_INFO(this->get_logger(), "Created session folder: %s", session_folder_.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "Session folder already exists or using existing directory: %s", session_folder_.c_str());
            }
        } catch (const std::filesystem::filesystem_error& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create session folder '%s': %s", session_folder_.c_str(), e.what());
            // Fallback to current directory without session subfolder? Or handle error appropriately.
            // For now, we'll proceed assuming the path might become valid later, but saving might fail.
            session_folder_ = session_name; // Use relative path if creation failed
             RCLCPP_WARN(this->get_logger(), "Falling back to relative session path: %s", session_folder_.c_str());
        }
    }

    // --- Visibility Check ---
    bool isCellInBounds(int x, int y, int width, int height) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    bool isMapOccupied(int x, int y, int width) {
        int idx = y * width + x;
        if (static_cast<size_t>(idx) >= base_map_.data.size()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "isVisible check: Index out of bounds.");
            return true; // Treat out of bounds as occupied/unpassable
        }
        return base_map_.data[idx] >= 50; // Occupied cell threshold
    }

    bool isVisible(int x0, int y0, int x1, int y1)
    {
        // Pre-check: If map isn't valid, assume not visible
        if (base_map_.info.width == 0 || base_map_.info.height == 0) return false;

        int width = base_map_.info.width;
        int height = base_map_.info.height;

        int dx = std::abs(x1 - x0);
        int dy = -std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx + dy;

        while (true)
        {
            if (!isCellInBounds(x0, y0, width, height)) return false; // Out of map bounds
            if (isMapOccupied(x0, y0, width)) return false;         // Hit obstacle
            if (x0 == x1 && y0 == y1) break;                       // Reached target

            int e2 = 2 * err;
            // Bresenham line algorithm steps
            if (e2 >= dy) { err += dy; x0 += sx; } // Step x
            if (e2 <= dx) { err += dx; y0 += sy; } // Step y
        }
        return true; // Reached target without hitting obstacle or going out of bounds
    }

    // --- Callbacks ---
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        if (msg->info.width == 0 || msg->info.height == 0) {
            RCLCPP_WARN(this->get_logger(), "Received map with zero dimensions. Ignoring.");
            return;
        }

        bool dimensions_changed = (rad_costmap_.info.width != msg->info.width ||
                                   rad_costmap_.info.height != msg->info.height);
        bool rad_field_needs_init = (rad_field_.empty() ||
                                      static_cast<unsigned int>(rad_field_.rows) != msg->info.height ||
                                      static_cast<unsigned int>(rad_field_.cols) != msg->info.width);

        base_map_ = *msg; // Update base map

        // Reinitialize structures if dimensions changed or rad_field is invalid
        if (dimensions_changed || rad_field_needs_init) {
             RCLCPP_INFO(this->get_logger(), "Map received/changed. Initializing/Resizing structures for size %u x %u", msg->info.width, msg->info.height);
             rad_costmap_.header = msg->header;
             rad_costmap_.info = msg->info;
             rad_costmap_.data.assign(msg->info.width * msg->info.height, -1); // Use -1 for unknown
             rad_field_ = cv::Mat::zeros(msg->info.height, msg->info.width, CV_32FC1);
        }
         // If only the map data updated but dimensions are the same, we don't need to reinitialize rad_field/rad_costmap
    }

    void radCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        // Basic validation of incoming message
        if (msg->data.empty() || msg->data[0] < min_valid_cpm_) {
            // RCLCPP_DEBUG(this->get_logger(), "Ignoring low CPM sample: %.2f", msg->data.empty() ? 0.0 : msg->data[0]);
            return;
        }

        geometry_msgs::msg::PoseStamped pose_in, pose_out;
        pose_in.header.frame_id = sensor_frame_;
        pose_in.header.stamp = this->now(); // Use current time for transform lookup

        // Attempt to transform sensor pose to map frame
        try {
            pose_out = tf_buffer_.transform(pose_in, map_frame_, tf2::durationFromSec(0.1));
        } catch (const tf2::TransformException &ex) {
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "TF Error transforming %s to %s: %s",
                                  sensor_frame_.c_str(), map_frame_.c_str(), ex.what());
            return;
        }

        double current_cpm = msg->data[0];

        // Lock map data for processing
        std::lock_guard<std::mutex> lock(map_mutex_);

        // Ensure map and internal structures are ready
         if (base_map_.info.width == 0 || base_map_.info.height == 0 || rad_field_.empty()) {
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Map/RadField not initialized. Skipping radiation sample.");
             return;
         }

        // Update overall maximum CPM encountered
        if (current_cpm > overall_max_cpm_) {
            overall_max_cpm_ = current_cpm;
            RCLCPP_INFO(this->get_logger(), "New overall max CPM: %.2f", overall_max_cpm_);
        }

        // Add the new sample and trigger field update
        samples_.push_back({pose_out.pose.position.x, pose_out.pose.position.y, current_cpm});
        updateRadiationField();
    }

    // --- Core Logic: Heatmap Update ---

    // Helper to calculate contribution of samples to a single cell (reduces nesting)
    std::pair<double, double> calculateCellContribution(int x, int y, double wx, double wy, double res,
                                                       double origin_x, double origin_y, double inv_2sigma2,
                                                       double distance_cutoff_sq, int grid_width, int grid_height)
    {
        double numerator = 0.0;
        double denominator = 0.0;

        for (const auto &sample : samples_) // Loop level 1
        {
            // Condition 1 (Level 2)
            if (sample.cpm < min_sample_cpm_) continue;

            int sx = static_cast<int>((sample.x - origin_x) / res);
            int sy = static_cast<int>((sample.y - origin_y) / res);

            // Condition 2 (Level 2) - Sample location valid?
            if (!isCellInBounds(sx, sy, grid_width, grid_height)) continue;

            // Condition 3 (Level 2) - Visible from sample?
            if (!isVisible(sx, sy, x, y)) continue;

            double dx = wx - sample.x;
            double dy = wy - sample.y;
            double dist_sq = dx * dx + dy * dy;

            // Condition 4 (Level 2) - Within distance cutoff?
            if (dist_sq > distance_cutoff_sq) continue;

            // Passed all checks, calculate weight and add contribution
            double weight = std::exp(-dist_sq * inv_2sigma2);
            numerator += weight * sample.cpm;
            denominator += weight;
        }
        return {numerator, denominator};
    }

    // Calculates the raw radiation field based on samples
    cv::Mat computeRawRadiationField() {
        cv::Mat current_rad_field = cv::Mat::zeros(base_map_.info.height, base_map_.info.width, CV_32FC1);
        double res = base_map_.info.resolution;
        double origin_x = base_map_.info.origin.position.x;
        double origin_y = base_map_.info.origin.position.y;
        double inv_2sigma2 = 1.0 / (2.0 * gaussian_sigma_ * gaussian_sigma_);
        double distance_cutoff_sq = distance_cutoff_ * distance_cutoff_;

        for (int y = 0; y < current_rad_field.rows; ++y) { // Loop level 1
            for (int x = 0; x < current_rad_field.cols; ++x) { // Loop level 2
                double wx = origin_x + x * res + res / 2.0;
                double wy = origin_y + y * res + res / 2.0;

                // Call helper function to get contributions (max nesting inside helper is level 2)
                auto [numerator, denominator] = calculateCellContribution(
                    x, y, wx, wy, res, origin_x, origin_y, inv_2sigma2, distance_cutoff_sq,
                    current_rad_field.cols, current_rad_field.rows);

                // Assign value to cell (avoid division by zero)
                current_rad_field.at<float>(y, x) = (denominator > 1e-9) ? static_cast<float>(numerator / denominator) : 0.0f;
            }
        }
        return current_rad_field;
    }

    // Applies Gaussian blur to the field
    void applyGaussianBlur(cv::Mat& field) {
        if (gaussian_blur_size_ < 3 || gaussian_sigma_ <= 0) return; // No blur needed
        cv::GaussianBlur(field, field, cv::Size(gaussian_blur_size_, gaussian_blur_size_), gaussian_sigma_);
    }

    // Applies background thresholding and finds true min/max CPM above threshold
    std::pair<float, float> applyThresholdAndGetRange(cv::Mat& field) {
        float true_min = std::numeric_limits<float>::max();
        float true_max = std::numeric_limits<float>::lowest();
        bool found_value_above_threshold = false;

        for (int y = 0; y < field.rows; ++y) { // Loop level 1
            for (int x = 0; x < field.cols; ++x) { // Loop level 2
                float &val = field.at<float>(y, x);
                if (val < background_threshold_) { // Condition level 3
                    val = 0.0f; // Set below threshold to 0
                } else {
                    true_min = std::min(true_min, val);
                    true_max = std::max(true_max, val);
                    found_value_above_threshold = true;
                }
            }
        }

        // Handle cases where no values are above threshold or range is zero
        if (!found_value_above_threshold) {
             if (!samples_.empty()) { // Only warn if we actually had samples
                 RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                      "No radiation values above background threshold: %.2f", background_threshold_);
             }
             true_min = background_threshold_; // Default range if nothing found
             true_max = background_threshold_ + 1.0f; // Avoid zero range
        } else if (true_max <= true_min) {
             true_max = true_min + 1.0f; // Ensure max > min
        }
        return {true_min, true_max};
    }

    // Updates the costmap based on the processed radiation field
    void updateCostmapData(const cv::Mat& field, float log_min, float log_range) {
        if (log_range <= 1e-6) { // Check for invalid range before loop
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,"Log range too small. Check thresholds/scaling.");
             log_range = 1.0f; // Prevent division by zero / large values
        }

        for (int y = 0; y < field.rows; ++y) { // Loop level 1
            for (int x = 0; x < field.cols; ++x) { // Loop level 2
                int idx = y * field.cols + x;
                if (idx < 0 || static_cast<size_t>(idx) >= rad_costmap_.data.size()) continue; // Bounds check

                float val = field.at<float>(y, x);
                if (val < background_threshold_) { // Condition level 3
                    rad_costmap_.data[idx] = -1; // Use -1 for unknown/no significant radiation
                } else {
                    float log_val = std::log(std::max(val, 1.0f)); // Use log scale, ensure arg >= 1
                    float scaled = 100.0f * (log_val - log_min) / log_range;
                    // Clamp and bucket the value (optional, but can make costmap interpretation easier)
                    // int8_t bucketed = static_cast<int8_t>(std::round(std::clamp(scaled, 0.0f, 100.0f) / 5.0f) * 5.0f);
                    // rad_costmap_.data[idx] = std::max(static_cast<int8_t>(0), std::min(bucketed, static_cast<int8_t>(100)));
                    rad_costmap_.data[idx] = static_cast<int8_t>(std::clamp(scaled, 0.0f, 100.0f)); // Direct scale 0-100
                }
            }
        }
    }

    // Creates the base map image (grayscale)
    cv::Mat createBaseMapImage() {
        cv::Mat base_gray(rad_field_.rows, rad_field_.cols, CV_8UC1);
        for (int y = 0; y < rad_field_.rows; ++y) { // Loop level 1
            for (int x = 0; x < rad_field_.cols; ++x) { // Loop level 2
                int idx = y * rad_field_.cols + x;
                uchar pixel_val = 127; // Default to gray for unknown/out-of-bounds
                if (isCellInBounds(x, y, base_map_.info.width, base_map_.info.height) &&
                    static_cast<size_t>(idx) < base_map_.data.size())
                { // Condition level 3
                    int8_t occ = base_map_.data[idx];
                    pixel_val = (occ == -1) ? 127 : (occ >= 50 ? 0 : 255); // Unknown=gray, Occupied=black, Free=white
                }
                base_gray.at<uchar>(y, x) = pixel_val;
            }
        }
        cv::Mat base_bgr;
        cv::cvtColor(base_gray, base_bgr, cv::COLOR_GRAY2BGR);
        return base_bgr;
    }

    // Creates the colored heatmap overlay image
    cv::Mat createColorHeatmapImage(const cv::Mat& field, float log_min, float log_range) {
         if (log_range <= 1e-6) log_range = 1.0f; // Prevent division issues

        cv::Mat scaled_field = cv::Mat::zeros(field.size(), CV_8UC1);
        for (int y = 0; y < field.rows; ++y) { // Loop level 1
            for (int x = 0; x < field.cols; ++x) { // Loop level 2
                float val = field.at<float>(y, x);
                if (val >= background_threshold_) { // Condition level 3
                    float log_val = std::log(std::max(val, 1.0f));
                    float scaled = 255.0f * (log_val - log_min) / log_range;
                    scaled_field.at<uchar>(y, x) = static_cast<uchar>(std::clamp(scaled, 0.0f, 255.0f));
                }
            }
        }

        cv::Mat color_field;
        cv::applyColorMap(scaled_field, color_field, cv::COLORMAP_JET);

        // Mask out areas below threshold (make them transparent/black in the overlay)
        cv::Mat mask;
        cv::compare(field, background_threshold_, mask, cv::CMP_LT); // Find pixels < threshold
        color_field.setTo(cv::Scalar(0, 0, 0), mask); // Set those pixels to black in the color map

        return color_field;
    }

    // Adds a color legend to the blended image
    void addLegendToImage(cv::Mat& blended_image, float current_min_cpm, float overall_max_cpm_used) {
         if (blended_image.empty()) return;

         int legend_width = 50;
         int legend_height = blended_image.rows / 2; // Adjust size as needed
         int legend_y = blended_image.rows / 4;      // Position
         int border_width = legend_width + 70;       // Space for legend + text

         // Add border to the right for the legend
         cv::copyMakeBorder(blended_image, blended_image, 0, 0, 0, border_width, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
         int legend_x = blended_image.cols - border_width + 10; // X position within the new border

         // Create the legend strip
         cv::Mat legend(legend_height, legend_width, CV_8UC3);
         for (int i = 0; i < legend_height; ++i) { // Loop level 1
             // Map legend strip value (0-255) to colormap
             uchar val_map = 255 - static_cast<uchar>((static_cast<float>(i) / legend_height) * 255.0f);
             cv::Mat color_val(1, 1, CV_8UC1, cv::Scalar(val_map));
             cv::Mat mapped_color;
             cv::applyColorMap(color_val, mapped_color, cv::COLORMAP_JET);
             if (!mapped_color.empty()) { // Condition level 2
                 legend.row(i) = mapped_color.at<cv::Vec3b>(0, 0);
             }
         }

         // Copy legend onto the bordered image if ROI is valid
         if (legend_x >= 0 && legend_y >= 0 && legend_x + legend_width <= blended_image.cols && legend_y + legend_height <= blended_image.rows) { // Condition level 1
              legend.copyTo(blended_image(cv::Rect(legend_x, legend_y, legend_width, legend_height)));
         } else {
              RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Legend ROI calculation is outside bordered image bounds.");
              return; // Don't draw text if legend failed
         }


         // Add text labels next to the legend
         int text_x = legend_x + legend_width + 5;
         if (text_x < blended_image.cols) { // Condition level 1
             cv::putText(blended_image, cv::format("%.1f", overall_max_cpm_used), cv::Point(text_x, legend_y + 15),
                         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
             cv::putText(blended_image, cv::format("%.1f", current_min_cpm), cv::Point(text_x, legend_y + legend_height - 5),
                         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
             cv::putText(blended_image, "CPM", cv::Point(text_x, legend_y + legend_height / 2),
                         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
         }
    }

    // Publishes the final costmap and image
    void publishResults(const cv::Mat& blended_image) {
        // Publish Costmap
        rad_costmap_.header.stamp = this->now();
        rad_costmap_.header.frame_id = map_frame_; // Ensure frame_id is set
        rad_costmap_pub_->publish(rad_costmap_);

        // Publish Image
        if (!blended_image.empty() && rad_image_pub_.getNumSubscribers() > 0) { // Condition level 1
             std_msgs::msg::Header header;
             header.stamp = this->now();
             header.frame_id = map_frame_; // Use map frame for the image too
             try { // Try block level 2
                 sensor_msgs::msg::Image::SharedPtr img_msg =
                     cv_bridge::CvImage(header, "bgr8", blended_image).toImageMsg();
                 rad_image_pub_.publish(*img_msg);
             } catch (const cv_bridge::Exception& e) { // Catch block level 2
                  RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "cv_bridge exception: %s", e.what());
             } catch (const cv::Exception& e) { // Catch block level 2
                  RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "OpenCV exception during image publishing: %s", e.what());
             }
        } else if (blended_image.empty()){
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Blended image is empty, cannot publish.");
        }
    }


    void updateRadiationField()
    {
        // Pre-condition check
        if (base_map_.info.width == 0 || base_map_.info.height == 0 || rad_field_.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "updateRadiationField called before map/rad_field initialized.");
            return;
        }

        // 1. Compute raw field from samples
        rad_field_ = computeRawRadiationField();

        // 2. Apply Gaussian Blur
        applyGaussianBlur(rad_field_);

        // 3. Apply background threshold and get min/max range above threshold
        auto [true_min_cpm, true_max_cpm] = applyThresholdAndGetRange(rad_field_);

        // 4. Calculate log scale range for coloring/costmap
        //    Use overall_max_cpm_ for the upper bound of the scale if it's higher
        //    Use true_min_cpm (post-thresholding) for the lower bound
        float scale_max_cpm = std::max(true_max_cpm, static_cast<float>(overall_max_cpm_));
        float log_min = std::log(std::max(true_min_cpm, 1.0f));
        float log_max = std::log(std::max(scale_max_cpm, true_min_cpm + 1.0f)); // Ensure max > min
        float log_range = log_max - log_min;

        // 5. Update the OccupancyGrid (Costmap) data
        updateCostmapData(rad_field_, log_min, log_range);

        // 6. Create visualization image: Base map + Heatmap overlay + Legend
        cv::Mat base_bgr = createBaseMapImage();
        cv::Mat color_field = createColorHeatmapImage(rad_field_, log_min, log_range);

        // Blend base map and heatmap
        cv::Mat blended_image;
        if (base_bgr.size() == color_field.size() && !base_bgr.empty()) {
             cv::addWeighted(base_bgr, 0.5, color_field, 0.5, 0.0, blended_image);
        } else {
             RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Size mismatch or empty image before blending.");
             return; // Cannot proceed if blending failed
        }


        // 7. Add legend
        addLegendToImage(blended_image, true_min_cpm, scale_max_cpm); // Use the actual range used for scaling

        // Store the latest image for saving on shutdown
        latest_blended_image_ = blended_image.clone();

        // 8. Publish costmap and image
        publishResults(blended_image);

    }

    // --- Shutdown ---
    void saveMap()
    {
        std::lock_guard<std::mutex> lock(map_mutex_); // Ensure thread safety if called concurrently

        // Check if a valid image was generated and session folder is set
        if (latest_blended_image_.empty()) {
             RCLCPP_WARN(this->get_logger(), "No valid heatmap image was generated to save.");
             return;
        }
         if (session_folder_.empty()) {
             RCLCPP_ERROR(this->get_logger(), "Session folder path is not set. Cannot save map.");
             return;
         }

        std::string output_path = session_folder_ + "/map_with_heatmap.png";

        try {
            // Ensure the directory exists one last time
            std::filesystem::path dir_path = std::filesystem::path(session_folder_);
            if (!std::filesystem::exists(dir_path)) {
                RCLCPP_WARN(this->get_logger(), "Session directory '%s' doesn't exist. Attempting to create.", session_folder_.c_str());
                std::filesystem::create_directories(dir_path); // Try creating again
            }

            bool success = cv::imwrite(output_path, latest_blended_image_);
            if (!success) {
                 RCLCPP_ERROR(this->get_logger(), "Failed to save heatmap image to: %s", output_path.c_str());
            } else {
                 RCLCPP_INFO(this->get_logger(), "Saved final heatmap image to: %s", output_path.c_str());
            }
        } catch (const std::filesystem::filesystem_error& e) {
             RCLCPP_ERROR(this->get_logger(), "Filesystem error during saveMap for path '%s': %s", output_path.c_str(), e.what());
        } catch (const cv::Exception& e) {
             RCLCPP_ERROR(this->get_logger(), "OpenCV Exception during saveMap: %s", e.what());
        }
    }


public:
    RadiationHeatmapNode()
        : Node("radiation_heatmap_node"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_),
          overall_max_cpm_(0.0) // Initialize overall max CPM
    {
        declareAndGetParameters();
        overall_max_cpm_ = background_threshold_; // Initialize overall max to background threshold initially
        createSessionFolder();
        setupCommunications();

        RCLCPP_INFO(this->get_logger(), "Radiation Heatmap Node Initialized.");
        RCLCPP_INFO(this->get_logger(), "Outputting artifacts to: %s", session_folder_.c_str());
        RCLCPP_INFO(this->get_logger(), "Waiting for map on topic /map...");

        // Register shutdown hook
        rclcpp::on_shutdown([this]() {
            RCLCPP_INFO(this->get_logger(), "Shutdown requested. Saving final map...");
            saveMap();
            RCLCPP_INFO(this->get_logger(), "Radiation Heatmap Node shutting down.");
        });
    }

    // Needs to be called after construction because it uses shared_from_this()
    void init_image_transport()
    {
        image_transport::ImageTransport it(shared_from_this());
        rad_image_pub_ = it.advertise("/rad_heatmap_image", 1);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RadiationHeatmapNode>();
    node->init_image_transport(); // Initialize image transport separately
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}