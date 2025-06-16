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
#include "time.h"

#include <vector>
#include <mutex>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "yaml-cpp/yaml.h"

class RadiationHeatmapNode : public rclcpp::Node
{
private:
    struct RadiationSample
    {
        double x, y, cpm, mrem_hr;
    };

    struct LoggedRadiationEntry
    {
        rclcpp::Time timestamp;
        double raw_cpm = 0.0;
        double raw_mrem_hr = 0.0;
        double map_x = std::numeric_limits<double>::quiet_NaN();
        double map_y = std::numeric_limits<double>::quiet_NaN();
        std::string status = "Unprocessed";
        std::string containing_zone_label = "N/A";
        std::string closest_zone_label = "N/A";
    };

    struct ZoneOfInterest
    {
        std::string label;
        double map_x, map_y;
        double radius;
    };

    std::string session_folder_;
    std::string sensor_frame_, map_frame_;
    std::string output_directory_;
    double min_valid_cpm_, min_sample_cpm_;
    double gaussian_sigma_, distance_cutoff_, background_threshold_mrem_hr_;
    int gaussian_blur_size_;
    double overall_max_mrem_hr_;
    double tf_timeout_seconds_;

    std::string predefined_zones_file_path_;
    std::vector<ZoneOfInterest> zones_of_interest_;

    std::vector<RadiationSample> samples_;
    std::vector<LoggedRadiationEntry> logged_rad_entries_;

    nav_msgs::msg::OccupancyGrid base_map_, rad_costmap_;
    cv::Mat rad_field_, latest_blended_image_;
    std::mutex map_mutex_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rad_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr rad_costmap_pub_;
    image_transport::Publisher rad_image_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    void declareAndGetParameters()
    {
        this->declare_parameter<std::string>("sensor_frame", "base_link");
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<std::string>("output_directory", "");
        this->declare_parameter<double>("min_valid_cpm", 50.0);
        this->declare_parameter<double>("min_sample_cpm", 100.0);
        this->declare_parameter<double>("gaussian_sigma", 5.0);
        this->declare_parameter<int>("gaussian_blur_size", 15);
        this->declare_parameter<double>("distance_cutoff", 10.0);
        this->declare_parameter<double>("background_threshold_mrem_hr", 0.1);
        this->declare_parameter<std::string>("predefined_zones_file", "");
        this->declare_parameter<double>("tf_timeout_seconds", 0.2);

        sensor_frame_ = this->get_parameter("sensor_frame").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        output_directory_ = this->get_parameter("output_directory").as_string();
        min_valid_cpm_ = this->get_parameter("min_valid_cpm").as_double();
        min_sample_cpm_ = this->get_parameter("min_sample_cpm").as_double();
        gaussian_sigma_ = this->get_parameter("gaussian_sigma").as_double();
        gaussian_blur_size_ = this->get_parameter("gaussian_blur_size").as_int();
        distance_cutoff_ = this->get_parameter("distance_cutoff").as_double();
        background_threshold_mrem_hr_ = this->get_parameter("background_threshold_mrem_hr").as_double();
        predefined_zones_file_path_ = this->get_parameter("predefined_zones_file").as_string();
        tf_timeout_seconds_ = this->get_parameter("tf_timeout_seconds").as_double();

        if (gaussian_blur_size_ <= 0)
            gaussian_blur_size_ = 1;
        else if (gaussian_blur_size_ % 2 == 0)
            gaussian_blur_size_++;

        RCLCPP_INFO(this->get_logger(), "Predefined zones file path: '%s'", predefined_zones_file_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "TF transform timeout: %.2f seconds", tf_timeout_seconds_);
        RCLCPP_INFO(this->get_logger(), "Background threshold (mrem/hr): %.4f", background_threshold_mrem_hr_);
    }

    void loadPredefinedZones()
    {
        if (predefined_zones_file_path_.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Parameter 'predefined_zones_file' is empty. No predefined zones will be loaded.");
            return;
        }
        if (!std::filesystem::exists(predefined_zones_file_path_))
        {
            RCLCPP_ERROR(this->get_logger(), "Predefined zones file not found at: '%s'. No zones loaded.", predefined_zones_file_path_.c_str());
            return;
        }

        try
        {
            YAML::Node config = YAML::LoadFile(predefined_zones_file_path_);
            if (config["predefined_zones"])
            {
                for (const auto &zone_node : config["predefined_zones"])
                {
                    if (!zone_node["label"] || !zone_node["center_x"] || !zone_node["center_y"] || !zone_node["radius"])
                    {
                        RCLCPP_WARN(this->get_logger(), "Skipping malformed zone entry in '%s'. Missing one or more required fields (label, center_x, center_y, radius).", predefined_zones_file_path_.c_str());
                        continue;
                    }
                    ZoneOfInterest zoi;
                    zoi.label = zone_node["label"].as<std::string>();
                    zoi.map_x = zone_node["center_x"].as<double>();
                    zoi.map_y = zone_node["center_y"].as<double>();
                    zoi.radius = zone_node["radius"].as<double>();

                    zones_of_interest_.push_back(zoi);
                    RCLCPP_INFO(this->get_logger(), "Loaded predefined zone: '%s' at (%.2f, %.2f), Radius: %.2fm",
                                zoi.label.c_str(), zoi.map_x, zoi.map_y, zoi.radius);
                }
                if (zones_of_interest_.empty() && config["predefined_zones"].IsSequence() && config["predefined_zones"].size() > 0)
                {
                    RCLCPP_WARN(this->get_logger(), "'predefined_zones' key exists in '%s' but all entries might be malformed or no valid zones were parsed.", predefined_zones_file_path_.c_str());
                }
                else if (zones_of_interest_.empty())
                {
                    RCLCPP_INFO(this->get_logger(), "'predefined_zones' key in '%s' is empty or does not contain valid zone entries.", predefined_zones_file_path_.c_str());
                }
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Predefined zones file '%s' does not contain 'predefined_zones' key or is empty.", predefined_zones_file_path_.c_str());
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load or parse predefined zones file '%s': %s",
                         predefined_zones_file_path_.c_str(), e.what());
        }
    }

    void setupCommunications()
    {
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(), std::bind(&RadiationHeatmapNode::mapCallback, this, std::placeholders::_1));
        rad_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("/rad", 10, std::bind(&RadiationHeatmapNode::radCallback, this, std::placeholders::_1));
        rad_costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rad_heatmap", 10);
    }

    void createSessionFolder()
    {
        std::stringstream ss;
        std::time_t now = std::time(nullptr);
        ss << "rad_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
        std::string session_name = ss.str();
        std::filesystem::path base_path;
        if (output_directory_.empty())
        {
            base_path = std::filesystem::current_path();
        }
        else
        {
            base_path = output_directory_;
        }
        session_folder_ = (base_path / session_name).string();
        try
        {
            if (!std::filesystem::exists(session_folder_))
            {
                if (!std::filesystem::create_directories(session_folder_))
                {
                    session_folder_ = session_name;
                    if (!std::filesystem::exists(session_folder_))
                        std::filesystem::create_directory(session_folder_);
                }
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            session_folder_ = "rad_heatmap_default_session";
            try
            {
                if (!std::filesystem::exists(session_folder_))
                    std::filesystem::create_directory(session_folder_);
            }
            catch (const std::filesystem::filesystem_error &fe)
            {
                session_folder_ = "";
            }
        }
        if (!session_folder_.empty())
            RCLCPP_INFO(this->get_logger(), "Session folder: %s", session_folder_.c_str());
        else
            RCLCPP_ERROR(this->get_logger(), "Could not create session folder. Data saving disabled.");
    }

    bool isCellInBounds(int x, int y, int width, int height) const { return x >= 0 && x < width && y >= 0 && y < height; }
    bool isMapOccupied(int x, int y, int width)
    {
        int idx = y * width + x;
        if (static_cast<size_t>(idx) >= base_map_.data.size())
            return true;
        return base_map_.data[idx] >= 50;
    }
    bool isVisible(int x0, int y0, int x1, int y1)
    {
        if (base_map_.info.width == 0)
            return false;
        int w = base_map_.info.width, h = base_map_.info.height;
        int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2;
        while (true)
        {
            if (!isCellInBounds(x0, y0, w, h) || isMapOccupied(x0, y0, w))
                return false;
            if (x0 == x1 && y0 == y1)
                break;
            e2 = 2 * err;
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
        if (msg->info.width == 0 || msg->info.height == 0)
        {
            return;
        }
        bool dim_changed = (rad_costmap_.info.width != msg->info.width || rad_costmap_.info.height != msg->info.height);
        bool field_init = (rad_field_.empty() || (unsigned int)rad_field_.rows != msg->info.height || (unsigned int)rad_field_.cols != msg->info.width);
        base_map_ = *msg;
        if (dim_changed || field_init)
        {
            rad_costmap_.header = msg->header;
            rad_costmap_.info = msg->info;
            rad_costmap_.data.assign(msg->info.width * msg->info.height, -1);
            rad_field_ = cv::Mat::zeros(msg->info.height, msg->info.width, CV_32FC1);
            updateRadiationField();
        }
    }

    void radCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        LoggedRadiationEntry current_log_entry;
        current_log_entry.timestamp = this->now();

        if (msg->data.empty())
        {
            current_log_entry.status = "EmptyDataPayload";
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "/rad message received with empty data payload.");
            std::lock_guard<std::mutex> lock(map_mutex_);
            logged_rad_entries_.push_back(current_log_entry);
            return;
        }

        current_log_entry.raw_cpm = msg->data[0];
        current_log_entry.raw_mrem_hr = (msg->data.size() > 1) ? msg->data[1] : 0.0;
        if (msg->data.size() <= 1)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000, "Rad message data[1] (mrem/hr) not available. Defaulting to 0 for this sample.");
        }

        if (current_log_entry.raw_cpm < min_valid_cpm_)
        {
            current_log_entry.status = "BelowMinValidCPM";
            std::lock_guard<std::mutex> lock(map_mutex_);
            logged_rad_entries_.push_back(current_log_entry);
            return;
        }

        geometry_msgs::msg::PoseStamped pose_in, pose_out;
        pose_in.header.frame_id = sensor_frame_;
        pose_in.header.stamp = this->now();
        bool tf_successful = false;

        try
        {
            pose_out = tf_buffer_.transform(pose_in, map_frame_, tf2::durationFromSec(tf_timeout_seconds_));
            current_log_entry.map_x = pose_out.pose.position.x;
            current_log_entry.map_y = pose_out.pose.position.y;
            tf_successful = true;
        }
        catch (const tf2::TransformException &ex)
        {
            current_log_entry.status = "TF_Failed";
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "TF Error transforming %s to %s: %s. Sample logged, not processed for heatmap.",
                                 sensor_frame_.c_str(), map_frame_.c_str(), ex.what());
        }

        std::lock_guard<std::mutex> lock(map_mutex_);

        bool process_for_heatmap = false;

        if (tf_successful)
        {
            if (!zones_of_interest_.empty())
            {
                double min_dist_sq = std::numeric_limits<double>::max();
                for (const auto &zone : zones_of_interest_)
                {
                    double dx = current_log_entry.map_x - zone.map_x;
                    double dy = current_log_entry.map_y - zone.map_y;
                    double dist_sq = dx * dx + dy * dy;

                    if (dist_sq < min_dist_sq)
                    {
                        min_dist_sq = dist_sq;
                        current_log_entry.closest_zone_label = zone.label;
                    }
                    if (dist_sq <= (zone.radius * zone.radius))
                    {
                        if (current_log_entry.containing_zone_label == "N/A")
                        {
                            current_log_entry.containing_zone_label = zone.label;
                        }
                    }
                }
            }

            if (current_log_entry.raw_cpm >= min_sample_cpm_)
            {
                current_log_entry.status = "ProcessedForHeatmap";
                double value_for_heatmap = (msg->data.size() > 1) ? current_log_entry.raw_mrem_hr : current_log_entry.raw_cpm;
                std::string unit_for_heatmap = (msg->data.size() > 1) ? "mrem/hr" : "CPM";

                samples_.push_back({current_log_entry.map_x, current_log_entry.map_y, current_log_entry.raw_cpm, current_log_entry.raw_mrem_hr});
                process_for_heatmap = true;

                if (current_log_entry.raw_mrem_hr > overall_max_mrem_hr_)
                {
                    overall_max_mrem_hr_ = current_log_entry.raw_mrem_hr;
                    RCLCPP_DEBUG(this->get_logger(), "New overall max mrem/hr for heatmap scaling: %.4f", overall_max_mrem_hr_);
                }
            }
            else
            {
                current_log_entry.status = "BelowMinSampleCPM_NotForHeatmap";
            }
        }

        logged_rad_entries_.push_back(current_log_entry);

        if (process_for_heatmap)
        {
            if (base_map_.info.width > 0 && base_map_.info.height > 0 && !rad_field_.empty())
            {
                updateRadiationField();
            }
            else
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Map/RadField not initialized. Skipping heatmap update for processed sample.");
            }
        }
    }

    std::pair<double, double> calculateCellContribution(int x_cell, int y_cell, double wx_cell_center, double wy_cell_center, double res,
                                                        double origin_x, double origin_y, double inv_2sigma2,
                                                        double distance_cutoff_sq, int grid_width, int grid_height)
    {
        double numerator = 0.0;
        double denominator = 0.0;
        for (const auto &sample : samples_)
        {
            int sx_sample_cell = static_cast<int>((sample.x - origin_x) / res);
            int sy_sample_cell = static_cast<int>((sample.y - origin_y) / res);
            if (!isCellInBounds(sx_sample_cell, sy_sample_cell, grid_width, grid_height))
                continue;
            if (!isVisible(sx_sample_cell, sy_sample_cell, x_cell, y_cell))
                continue;
            double dx_world = wx_cell_center - sample.x;
            double dy_world = wy_cell_center - sample.y;
            double dist_sq_world = dx_world * dx_world + dy_world * dy_world;
            if (dist_sq_world > distance_cutoff_sq)
                continue;
            double weight = std::exp(-dist_sq_world * inv_2sigma2);
            numerator += weight * sample.mrem_hr;
            denominator += weight;
        }
        return {numerator, denominator};
    }

    cv::Mat computeRawRadiationField()
    {
        cv::Mat current_rad_field = cv::Mat::zeros(base_map_.info.height, base_map_.info.width, CV_32FC1);
        double res = base_map_.info.resolution;
        double origin_x = base_map_.info.origin.position.x;
        double origin_y = base_map_.info.origin.position.y;
        double inv_2sigma2 = 1.0 / (2.0 * gaussian_sigma_ * gaussian_sigma_);
        double distance_cutoff_sq = distance_cutoff_ * distance_cutoff_;
        for (int y = 0; y < current_rad_field.rows; ++y)
        {
            for (int x = 0; x < current_rad_field.cols; ++x)
            {
                double wx_center = origin_x + (x + 0.5) * res;
                double wy_center = origin_y + (y + 0.5) * res;
                auto [numerator, denominator] = calculateCellContribution(
                    x, y, wx_center, wy_center, res, origin_x, origin_y, inv_2sigma2, distance_cutoff_sq,
                    current_rad_field.cols, current_rad_field.rows);
                current_rad_field.at<float>(y, x) = (denominator > 1e-9) ? static_cast<float>(numerator / denominator) : 0.0f;
            }
        }
        return current_rad_field;
    }

    void applyGaussianBlur(cv::Mat &field)
    {
        if (gaussian_blur_size_ < 3 || gaussian_sigma_ <= 0)
            return;
        cv::GaussianBlur(field, field, cv::Size(gaussian_blur_size_, gaussian_blur_size_), gaussian_sigma_);
    }

    std::pair<float, float> applyThresholdAndGetRange(cv::Mat &field)
    {
        float true_min = std::numeric_limits<float>::max();
        float true_max = std::numeric_limits<float>::lowest();
        bool found_value_above_threshold = false;
        for (int y = 0; y < field.rows; ++y)
        {
            for (int x = 0; x < field.cols; ++x)
            {
                float &val = field.at<float>(y, x);
                if (val < background_threshold_mrem_hr_)
                {
                    val = 0.0f;
                }
                else
                {
                    true_min = std::min(true_min, val);
                    true_max = std::max(true_max, val);
                    found_value_above_threshold = true;
                }
            }
        }
        if (!found_value_above_threshold)
        {
            true_min = static_cast<float>(background_threshold_mrem_hr_);
            true_max = static_cast<float>(background_threshold_mrem_hr_ + 0.01f);
            if (!samples_.empty())
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                     "No radiation values (mrem/hr) above background threshold: %.4f. Heatmap might appear blank.", background_threshold_mrem_hr_);
            }
        }
        else if (true_max <= true_min)
        {
            true_max = true_min + 0.01f;
        }
        return {true_min, true_max};
    }

    void updateCostmapData(const cv::Mat &field, float current_field_min_val, float current_field_max_val)
    {
        float log_min_val = std::log(std::max(current_field_min_val, 0.001f));
        float log_max_val = std::log(std::max(current_field_max_val, current_field_min_val + 0.01f));
        float log_range = log_max_val - log_min_val;

        if (log_range <= 1e-6)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000, "Logarithmic range for costmap (mrem/hr) is too small. Costmap might not be representative.");
            log_range = 1.0f;
        }

        for (int y = 0; y < field.rows; ++y)
        {
            for (int x = 0; x < field.cols; ++x)
            {
                int idx = y * field.cols + x;
                if (idx < 0 || static_cast<size_t>(idx) >= rad_costmap_.data.size())
                    continue;

                float val = field.at<float>(y, x);
                if (val < background_threshold_mrem_hr_)
                {
                    rad_costmap_.data[idx] = -1;
                }
                else
                {
                    float log_val = std::log(std::max(val, 0.001f));
                    float scaled_cost = 100.0f * (log_val - log_min_val) / log_range;
                    rad_costmap_.data[idx] = static_cast<int8_t>(std::clamp(scaled_cost, 0.0f, 100.0f));
                }
            }
        }
    }

    cv::Mat createBaseMapImage()
    {
        cv::Mat base_gray(rad_field_.rows, rad_field_.cols, CV_8UC1);
        for (int y = 0; y < rad_field_.rows; ++y)
            for (int x = 0; x < rad_field_.cols; ++x)
            {
                int idx = y * rad_field_.cols + x;
                uchar pv = 127; // Default to unknown
                if (isCellInBounds(x, y, base_map_.info.width, base_map_.info.height) && static_cast<size_t>(idx) < base_map_.data.size())
                {
                    int8_t o = base_map_.data[idx];
                    pv = (o == -1) ? 127 : (o >= 50 ? 0 : 255); // unknown, occupied, free
                }
                base_gray.at<uchar>(y, x) = pv; // Corrected line
            }
        cv::Mat bgr;
        cv::cvtColor(base_gray, bgr, cv::COLOR_GRAY2BGR); // Corrected line
        return bgr;
    }


    cv::Mat createColorHeatmapImage(const cv::Mat &field, float current_field_min_val, float current_field_max_val)
    {
        float log_min_val = std::log(std::max(current_field_min_val, 0.001f));
        float log_max_val = std::log(std::max(current_field_max_val, current_field_min_val + 0.01f));
        float log_range = log_max_val - log_min_val;
        if (log_range <= 1e-6)
            log_range = 1.0f;

        cv::Mat scaled_field_for_colormap = cv::Mat::zeros(field.size(), CV_8UC1);
        for (int y = 0; y < field.rows; ++y)
        {
            for (int x = 0; x < field.cols; ++x)
            {
                float val = field.at<float>(y, x);
                if (val >= background_threshold_mrem_hr_)
                {
                    float log_val = std::log(std::max(val, 0.001f));
                    float scaled_intensity = 255.0f * (log_val - log_min_val) / log_range;
                    scaled_field_for_colormap.at<uchar>(y, x) = static_cast<uchar>(std::clamp(scaled_intensity, 0.0f, 255.0f));
                }
            }
        }
        cv::Mat color_field;
        cv::applyColorMap(scaled_field_for_colormap, color_field, cv::COLORMAP_JET);
        cv::Mat mask_below_threshold;
        cv::compare(field, background_threshold_mrem_hr_, mask_below_threshold, cv::CMP_LT);
        color_field.setTo(cv::Scalar(0, 0, 0), mask_below_threshold);
        return color_field;
    }

    void addLegendToImage(cv::Mat &blended_image, float display_min_val, float display_max_val)
    {
        if (blended_image.empty())
            return;
        int legend_width = 50;
        int legend_height = blended_image.rows / 2;
        int legend_y_offset = blended_image.rows / 4;
        int border_padding = 80;
        int total_legend_area_width = legend_width + border_padding;
        cv::copyMakeBorder(blended_image, blended_image, 0, 0, 0, total_legend_area_width, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        int legend_origin_x = blended_image.cols - total_legend_area_width + 5;
        int legend_origin_y = legend_y_offset;
        cv::Mat legend_bar(legend_height, legend_width, CV_8UC3);
        for (int i = 0; i < legend_height; ++i)
        {
            uchar val_for_map = 255 - static_cast<uchar>((static_cast<float>(i) / legend_height) * 255.0f);
            cv::Mat single_color_value(1, 1, CV_8UC1, cv::Scalar(val_for_map));
            cv::Mat mapped_color_strip;
            cv::applyColorMap(single_color_value, mapped_color_strip, cv::COLORMAP_JET);
            if (!mapped_color_strip.empty())
            {
                legend_bar.row(i) = mapped_color_strip.at<cv::Vec3b>(0, 0);
            }
        }
        if (legend_origin_x >= 0 && legend_origin_y >= 0 &&
            legend_origin_x + legend_width <= blended_image.cols &&
            legend_origin_y + legend_height <= blended_image.rows)
        {
            legend_bar.copyTo(blended_image(cv::Rect(legend_origin_x, legend_origin_y, legend_width, legend_height)));
        }
        else
        {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Legend ROI calculation is outside bordered image bounds.");
            return;
        }
        int text_start_x = legend_origin_x + legend_width + 5;
        if (text_start_x < blended_image.cols)
        {
            cv::putText(blended_image, cv::format("%.2f", display_max_val), cv::Point(text_start_x, legend_origin_y + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            cv::putText(blended_image, cv::format("%.2f", display_min_val), cv::Point(text_start_x, legend_origin_y + legend_height - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            cv::putText(blended_image, "mrem/hr", cv::Point(text_start_x, legend_origin_y + legend_height / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }

    void drawZoneMarkers(cv::Mat &image_to_draw_on)
    {
        if (base_map_.info.width == 0 || base_map_.info.height == 0 || image_to_draw_on.empty())
            return;
        double res = base_map_.info.resolution;
        double ox = base_map_.info.origin.position.x;
        double oy = base_map_.info.origin.position.y;
        for (const auto &zone : zones_of_interest_)
        {
            int px = static_cast<int>((zone.map_x - ox) / res);
            int py = static_cast<int>((zone.map_y - oy) / res);
            int pr = static_cast<int>(zone.radius / res);
            if (px >= 0 && px < image_to_draw_on.cols && py >= 0 && py < image_to_draw_on.rows)
            {
                cv::Point c(px, py);
                cv::circle(image_to_draw_on, c, pr, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
                cv::circle(image_to_draw_on, c, 5, cv::Scalar(255, 255, 255), -1);
                cv::circle(image_to_draw_on, c, 5, cv::Scalar(0, 0, 0), 1);
                cv::Point to(px + 8, py - 8);
                if (to.x < 0)
                    to.x = 5;
                if (to.y < 15)
                    to.y = 15;
                if (to.x > image_to_draw_on.cols - 50)
                    to.x = image_to_draw_on.cols - 50;
                if (to.y > image_to_draw_on.rows - 5)
                    to.y = image_to_draw_on.rows - 5;
                cv::putText(image_to_draw_on, zone.label, to, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
                cv::putText(image_to_draw_on, zone.label, to, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }
    }

    void publishResults(const cv::Mat &blended_image_to_publish)
    {
        rad_costmap_.header.stamp = this->now();
        rad_costmap_.header.frame_id = map_frame_;
        rad_costmap_pub_->publish(rad_costmap_);
        if (!blended_image_to_publish.empty() && rad_image_pub_.getNumSubscribers() > 0)
        {
            std_msgs::msg::Header h;
            h.stamp = this->now();
            h.frame_id = map_frame_;
            try
            {
                rad_image_pub_.publish(*cv_bridge::CvImage(h, "bgr8", blended_image_to_publish).toImageMsg());
            }
            catch (const cv_bridge::Exception &e)
            {
                RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cv_bridge ex: %s", e.what());
            }
            catch (const cv::Exception &e)
            {
                RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "OpenCV ex: %s", e.what());
            }
        }
        else if (blended_image_to_publish.empty())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Blended image empty.");
        }
    }

    void updateRadiationField()
    {
        if (base_map_.info.width == 0 || base_map_.info.height == 0 || rad_field_.empty())
        {
            RCLCPP_WARN_ONCE(this->get_logger(), "updateRadiationField called before map/rad_field fully initialized. Skipping update.");
            return;
        }
        rad_field_ = computeRawRadiationField();
        applyGaussianBlur(rad_field_);
        auto [current_min_val_on_field, current_max_val_on_field] = applyThresholdAndGetRange(rad_field_);
        float display_scaling_max_val = std::max(current_max_val_on_field, static_cast<float>(overall_max_mrem_hr_));
        float display_scaling_min_val = current_min_val_on_field;
        updateCostmapData(rad_field_, display_scaling_min_val, display_scaling_max_val);

        cv::Mat base_bgr = createBaseMapImage();
        cv::Mat color_heatmap_overlay = createColorHeatmapImage(rad_field_, display_scaling_min_val, display_scaling_max_val);
        cv::Mat blended_image_output;
        if (base_bgr.size() == color_heatmap_overlay.size() && !base_bgr.empty())
        {
            cv::addWeighted(base_bgr, 0.5, color_heatmap_overlay, 0.5, 0.0, blended_image_output);
        }
        else
        {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Size mismatch or empty image before blending. Cannot update heatmap image.");
            if (!latest_blended_image_.empty())
            {
                publishResults(latest_blended_image_);
            }
            return;
        }
        addLegendToImage(blended_image_output, display_scaling_min_val, display_scaling_max_val);
        if (!zones_of_interest_.empty())
        {
            drawZoneMarkers(blended_image_output);
        }
        latest_blended_image_ = blended_image_output.clone();
        publishResults(blended_image_output);
    }

    void saveMap()
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (latest_blended_image_.empty() || session_folder_.empty())
        {
            RCLCPP_WARN(get_logger(), "No image/session folder to save map.");
            return;
        }
        std::string p = session_folder_ + "/map_with_heatmap.png";
        try
        {
            if (!std::filesystem::exists(session_folder_))
                std::filesystem::create_directories(session_folder_);
            if (!cv::imwrite(p, latest_blended_image_))
                RCLCPP_ERROR(get_logger(), "Failed to save %s", p.c_str());
            else
                RCLCPP_INFO(get_logger(), "Saved %s", p.c_str());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(get_logger(), "SaveMap ex: %s", e.what());
        }
    }

    void saveRadiationData()
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (logged_rad_entries_.empty())
        {
            RCLCPP_INFO(this->get_logger(), "No radiation data entries logged to save.");
            return;
        }
        if (session_folder_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Session folder path is not set or invalid. Cannot save radiation data CSV.");
            return;
        }
        std::string output_path = session_folder_ + "/radiation_data.csv";
        std::ofstream data_file(output_path);
        if (!data_file.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open radiation_data.csv for writing: %s", output_path.c_str());
            return;
        }
        data_file << "Timestamp_sec,Timestamp_nanosec,RawCPM,RawMRHR,MapX,MapY,Status,ContainingZoneLabel,ClosestZoneLabel\n";
        for (const auto &entry : logged_rad_entries_)
        {
            uint64_t total_nanoseconds = entry.timestamp.nanoseconds();
            uint32_t seconds = static_cast<uint32_t>(total_nanoseconds / 1000000000ULL);
            uint32_t nanoseconds_part = static_cast<uint32_t>(total_nanoseconds % 1000000000ULL);

            data_file << seconds << ","
                      << nanoseconds_part << ","
                      << entry.raw_cpm << ","
                      << entry.raw_mrem_hr << ",";
            if (std::isnan(entry.map_x))
                data_file << ",";
            else
                data_file << entry.map_x << ",";
            if (std::isnan(entry.map_y))
                data_file << ",";
            else
                data_file << entry.map_y << ",";
            data_file << "\"" << entry.status << "\",";
            data_file << "\"" << entry.containing_zone_label << "\",";
            data_file << "\"" << entry.closest_zone_label << "\"\n";
        }
        data_file.close();
        if (data_file.good())
        {
            RCLCPP_INFO(this->get_logger(), "Saved all %zu radiation log entries to: %s", logged_rad_entries_.size(), output_path.c_str());
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Error occurred while writing or closing radiation_data.csv: %s", output_path.c_str());
        }
    }

public:
    RadiationHeatmapNode()
        : Node("radiation_heatmap_node"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_),
          overall_max_mrem_hr_(0.0)
    {
        declareAndGetParameters();
        overall_max_mrem_hr_ = background_threshold_mrem_hr_;
        createSessionFolder();
        loadPredefinedZones();
        setupCommunications();

        RCLCPP_INFO(this->get_logger(), "Radiation Heatmap Node Initialized.");
        if (!session_folder_.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Outputting artifacts to: %s", session_folder_.c_str());
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Session folder could not be established. Data saving will be disabled.");
        }
        RCLCPP_INFO(this->get_logger(), "Listening for map on /map and radiation on /rad.");

        if (!predefined_zones_file_path_.empty() && zones_of_interest_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Predefined zones file was specified ('%s') but no zones were loaded. Check file content, path, and permissions.", predefined_zones_file_path_.c_str());
        }
        else if (!zones_of_interest_.empty())
        {
            RCLCPP_INFO(this->get_logger(), "%zu predefined zones loaded. Monitoring active.", zones_of_interest_.size());
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "No predefined zones specified or loaded. Heatmap will be generated without zone-specific averaging.");
        }

        rclcpp::on_shutdown([this]()
                            {
            RCLCPP_INFO(this->get_logger(), "Shutdown requested. Saving final map and radiation data...");
            saveMap();
            saveRadiationData();
            RCLCPP_INFO(this->get_logger(), "Radiation Heatmap Node shutting down gracefully."); });
    }

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
    node->init_image_transport();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
