#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem> // Requires C++17
#include <functional>
#include <iomanip> // For std::put_time
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream> // For formatting timestamp
#include <string>
#include <unordered_map>
#include <vector>

// ROS / TF2 Includes
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/context.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2/time.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

// PCL Includes
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Voxel Grid Data Structures ---
struct VoxelIndex
{
    int x, y, z;
    bool operator==(const VoxelIndex &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};
struct VoxelIndexHash
{
    std::size_t operator()(const VoxelIndex &vi) const {
        size_t h1 = std::hash<int>{}(vi.x);
        size_t h2 = std::hash<int>{}(vi.y);
        size_t h3 = std::hash<int>{}(vi.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
struct PointData
{
    float x, y, z;
    float intensity; // Absolute intensity
};
// --- End Voxel Grid Data Structures ---

class RadiationPointcloudNode : public rclcpp::Node
{
public:
    explicit RadiationPointcloudNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : Node("radiation_pointcloud_node", options),
          clock_(this->get_clock()),
          tf_buffer_(clock_),
          tf_listener_(tf_buffer_)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing RadiationPointCloudGenerator Node...");

        this->setupParameters();
        this->logParameters();
        this->validateParameters();

        global_max_absolute_intensity_ = min_intensity_publish_;

        this->setupROSInterfaces();

        // Register shutdown callback if enabled
        if (save_on_shutdown_) {
             // Simply call the function to register the callback, no assignment needed
             rclcpp::on_shutdown(
                 [this]() { this->onShutdownCallback(); }
             );
             RCLCPP_INFO(this->get_logger(), "Registered automatic PLY save on shutdown.");
        } else {
             RCLCPP_INFO(this->get_logger(), "Automatic PLY save on shutdown is disabled.");
        }
        RCLCPP_INFO(this->get_logger(), "Radiation PointCloud Generator Node Initialized.");
    }

private:
    // --- Parameter Handling ---
    void setupParameters()
    {
        this->declare_parameter<std::string>("radiation_topic", "/rad");
        this->declare_parameter<std::string>("lidar_topic", "/velodyne_points");
        this->declare_parameter<std::string>("output_topic", "/rad_pointcloud_accumulated");
        this->declare_parameter<std::string>("source_estimate_topic", "/rad_source_estimate");
        this->declare_parameter<std::string>("sensor_frame", "base_link");
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<double>("attenuation_coefficient_cm", 0.00012);
        this->declare_parameter<double>("min_cpm_threshold", 10.0);
        this->declare_parameter<double>("ray_max_distance", 5.0);
        this->declare_parameter<double>("min_intensity_publish", 1.0);
        this->declare_parameter<double>("voxel_resolution", 0.1);
        this->declare_parameter<double>("publish_rate", 1.0);
        this->declare_parameter<double>("tf_lookup_timeout", 0.15);
        this->declare_parameter<bool>("save_on_shutdown", true);
        this->declare_parameter<std::string>("save_ply_directory", "/tmp");
        this->declare_parameter<std::string>("save_ply_base_name", "rad_cloud");

        radiation_topic_ = this->get_parameter("radiation_topic").as_string();
        lidar_topic_ = this->get_parameter("lidar_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        source_estimate_topic_ = this->get_parameter("source_estimate_topic").as_string();
        sensor_frame_ = this->get_parameter("sensor_frame").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        attenuation_mu_ = this->get_parameter("attenuation_coefficient_cm").as_double() * 100.0;
        min_cpm_ = this->get_parameter("min_cpm_threshold").as_double();
        ray_max_distance_ = this->get_parameter("ray_max_distance").as_double();
        min_intensity_publish_ = this->get_parameter("min_intensity_publish").as_double();
        voxel_resolution_ = this->get_parameter("voxel_resolution").as_double();
        publish_rate_ = this->get_parameter("publish_rate").as_double();
        save_on_shutdown_ = this->get_parameter("save_on_shutdown").as_bool();
        save_ply_directory_ = this->get_parameter("save_ply_directory").as_string();
        save_ply_base_name_ = this->get_parameter("save_ply_base_name").as_string();

        double timeout_seconds = this->get_parameter("tf_lookup_timeout").as_double();
        tf_timeout_ = std::chrono::duration<double>(timeout_seconds);
    }

    void logParameters()
    {
        RCLCPP_INFO(this->get_logger(), "Parameters loaded:");
        RCLCPP_INFO(this->get_logger(), "  Radiation topic: %s", radiation_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  LiDAR topic: %s", lidar_topic_.c_str());
        // ... (rest of logging as before) ...
        RCLCPP_INFO(this->get_logger(), "  TF Lookup Timeout: %.2f s", tf_timeout_.count());
        RCLCPP_INFO(this->get_logger(), "  Save on Shutdown: %s", save_on_shutdown_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  Save PLY Directory: %s", save_ply_directory_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Save PLY Base Name: %s", save_ply_base_name_.c_str());
    }

    void validateParameters()
    {
         if (voxel_resolution_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(), "Voxel resolution must be positive, defaulting to 0.1m.");
            voxel_resolution_ = 0.1;
         }
         if (publish_rate_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(), "Publish rate must be positive, defaulting to 1.0 Hz.");
            publish_rate_ = 1.0;
         }
    }

    void setupROSInterfaces()
    {
        auto pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        accumulated_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, pointcloud_qos);
        source_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(source_estimate_topic_, 10);

        auto default_qos = rclcpp::SensorDataQoS();
        rad_sub_separate_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            radiation_topic_, default_qos, std::bind(&RadiationPointcloudNode::radCallback, this, std::placeholders::_1));
        lidar_sub_separate_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, default_qos, std::bind(&RadiationPointcloudNode::lidarCallback, this, std::placeholders::_1));

        publish_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_rate_),
            std::bind(&RadiationPointcloudNode::publishAccumulatedCloud, this));

        save_ply_service_ = this->create_service<std_srvs::srv::Trigger>(
            "~/save_ply",
            std::bind(&RadiationPointcloudNode::savePlyCallback, this,
                      std::placeholders::_1, std::placeholders::_2));
        RCLCPP_INFO(this->get_logger(), "Manual Save PLY service available at '~/save_ply'");
    }

    // --- Callbacks ---

    void radCallback(const std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
        if (msg->data.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "Received empty data array on %s topic.", radiation_topic_.c_str());
            return;
        }
        latest_cpm_ = msg->data[0];
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        double current_cpm = latest_cpm_;
        if (current_cpm < min_cpm_) { return; }
        if (msg->width * msg->height == 0) { return; }
        const std::string lidar_frame = msg->header.frame_id;
        if (lidar_frame.empty()) { return; }

        geometry_msgs::msg::TransformStamped T_sensor_lidar, T_map_lidar;
        rclcpp::Time target_time = msg->header.stamp;
        try {
             T_sensor_lidar = tf_buffer_.lookupTransform(sensor_frame_, lidar_frame, target_time, tf_timeout_);
             T_map_lidar = tf_buffer_.lookupTransform(map_frame_, lidar_frame, target_time, tf_timeout_);
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "TF lookup failed: %s. Skipping scan.", ex.what());
            return;
        }

        tf2::Vector3 P_sensor_in_lidar_frame(-T_sensor_lidar.transform.translation.x,
                                             -T_sensor_lidar.transform.translation.y,
                                             -T_sensor_lidar.transform.translation.z);
        std::map<uint32_t, float> current_scan_intensities;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        uint32_t current_lidar_idx = 0;

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++current_lidar_idx) {
            tf2::Vector3 P_point_in_lidar_frame(*iter_x, *iter_y, *iter_z);
            if (std::isnan(P_point_in_lidar_frame.x()) || std::isnan(P_point_in_lidar_frame.y()) || std::isnan(P_point_in_lidar_frame.z())) { continue; }

            tf2::Vector3 V_sensor_to_point = P_point_in_lidar_frame - P_sensor_in_lidar_frame;
            double dist = V_sensor_to_point.length();
            if (dist > ray_max_distance_ || dist < 1e-3) { continue; }

            double absolute_intensity = current_cpm * std::exp(-attenuation_mu_ * dist);

            if (absolute_intensity > global_max_absolute_intensity_) {
                global_max_absolute_intensity_ = absolute_intensity;
            }

            if (absolute_intensity < min_intensity_publish_) { continue; }
            current_scan_intensities[current_lidar_idx] = static_cast<float>(absolute_intensity);
        }

        if (current_scan_intensities.empty()){ return; }

        std::lock_guard<std::mutex> lock(voxel_map_mutex_);
        uint32_t current_lidar_idx_output = 0;
        sensor_msgs::PointCloud2ConstIterator<float> iter_lidar_x_final(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_lidar_y_final(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_lidar_z_final(*msg, "z");

        for (; iter_lidar_x_final != iter_lidar_x_final.end();
             ++iter_lidar_x_final, ++iter_lidar_y_final, ++iter_lidar_z_final, ++current_lidar_idx_output)
        {
            auto hit_it = current_scan_intensities.find(current_lidar_idx_output);
            if (hit_it == current_scan_intensities.end()) { continue; }

            geometry_msgs::msg::PointStamped p_lidar;
            p_lidar.header.frame_id = lidar_frame;
            p_lidar.header.stamp = target_time;
            double px = *iter_lidar_x_final; double py = *iter_lidar_y_final; double pz = *iter_lidar_z_final;
            if(std::isnan(px) || std::isnan(py) || std::isnan(pz)) { continue; }
            p_lidar.point.x = px; p_lidar.point.y = py; p_lidar.point.z = pz;

            geometry_msgs::msg::PointStamped p_map;
            try {
                tf2::doTransform(p_lidar, p_map, T_map_lidar);
            } catch (const tf2::TransformException &ex) {
                 RCLCPP_WARN(this->get_logger(), "Point transform failed index %u: %s. Skipping.", current_lidar_idx_output, ex.what());
                 continue;
            }

            float map_x = static_cast<float>(p_map.point.x);
            float map_y = static_cast<float>(p_map.point.y);
            float map_z = static_cast<float>(p_map.point.z);
            float current_intensity = hit_it->second;

            VoxelIndex vi;
            vi.x = static_cast<int>(std::floor(map_x / voxel_resolution_));
            vi.y = static_cast<int>(std::floor(map_y / voxel_resolution_));
            vi.z = static_cast<int>(std::floor(map_z / voxel_resolution_));

            auto map_entry = voxel_map_.find(vi);
            if (map_entry == voxel_map_.end() || current_intensity > map_entry->second.intensity) {
                voxel_map_[vi] = {map_x, map_y, map_z, current_intensity};
            }
        }
    }

    void publishAccumulatedCloud() {
        sensor_msgs::msg::PointCloud2 accumulated_msg;
        double sum_x_intensity = 0.0, sum_y_intensity = 0.0, sum_z_intensity = 0.0, sum_intensity = 0.0;

        {
            std::lock_guard<std::mutex> lock(voxel_map_mutex_);
            if (voxel_map_.empty()) {
                 this->publishEmptyCloud();
                 return;
            }
            setupPointCloud2Msg(accumulated_msg, voxel_map_.size());

            sensor_msgs::PointCloud2Iterator<float> iter_x(accumulated_msg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(accumulated_msg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(accumulated_msg, "z");
            sensor_msgs::PointCloud2Iterator<float> iter_intensity(accumulated_msg, "intensity");

            for (const auto& pair : voxel_map_) {
                const PointData& p_data = pair.second;
                if (std::isnan(p_data.x) || std::isnan(p_data.y) || std::isnan(p_data.z) || std::isnan(p_data.intensity)) {
                     accumulated_msg.is_dense = false;
                     ++iter_x; ++iter_y; ++iter_z; ++iter_intensity;
                     continue;
                 }
                *iter_x = p_data.x; *iter_y = p_data.y; *iter_z = p_data.z;
                *iter_intensity = p_data.intensity; // Store absolute intensity temporarily
                ++iter_x; ++iter_y; ++iter_z; ++iter_intensity;
            }
        }

        double min_norm = min_intensity_publish_;
        double max_norm = global_max_absolute_intensity_;
        double norm_range = max_norm - min_norm;
        if (norm_range <= 1e-6) { norm_range = 1.0; }

        sensor_msgs::PointCloud2Iterator<float> iter_norm_x(accumulated_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_norm_y(accumulated_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_norm_z(accumulated_msg, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_norm_intensity(accumulated_msg, "intensity");

        for (size_t i = 0; i < accumulated_msg.width; ++i) {
            float absolute_intensity = *iter_norm_intensity;
            double normalized_value = std::max(0.0, std::min(1.0, (absolute_intensity - min_norm) / norm_range));
            float output_intensity = static_cast<float>(normalized_value * 100.0);
            *iter_norm_intensity = output_intensity; // Overwrite with normalized [0-100]

            if (absolute_intensity > 1e-6) {
                 sum_x_intensity += (*iter_norm_x) * absolute_intensity;
                 sum_y_intensity += (*iter_norm_y) * absolute_intensity;
                 sum_z_intensity += (*iter_norm_z) * absolute_intensity;
                 sum_intensity += absolute_intensity;
            }
            ++iter_norm_x; ++iter_norm_y; ++iter_norm_z; ++iter_norm_intensity;
        }

        accumulated_pc_pub_->publish(accumulated_msg);
        if (sum_intensity > 1e-6) {
            publishCentroid(sum_x_intensity / sum_intensity, sum_y_intensity / sum_intensity, sum_z_intensity / sum_intensity, accumulated_msg.header.stamp);
        } else if (accumulated_msg.width > 0) {
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000, "Cannot calculate centroid: intensity sum %.4e too small.", sum_intensity);
        }
    }

    // --- PLY Saving Logic ---

    std::string getTimestampString() {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
        return ss.str();
    }

    bool savePlyToFile(const std::string& filepath) {
        RCLCPP_INFO(this->get_logger(), "Attempting to save PLY to: '%s'", filepath.c_str());
        if (filepath.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Provided filepath is empty.");
            return false;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        size_t map_size = 0;

        {
            std::lock_guard<std::mutex> lock(voxel_map_mutex_);
            map_size = voxel_map_.size();
            if (map_size == 0) {
                RCLCPP_WARN(this->get_logger(), "Voxel map is empty. Cannot save PLY file.");
                return false;
            }
            cloud->points.reserve(map_size);
            for (const auto& pair : voxel_map_) {
                const PointData& p_data = pair.second;
                if (std::isnan(p_data.x) || std::isnan(p_data.y) || std::isnan(p_data.z) || std::isnan(p_data.intensity)) {
                    continue; // Skip NaN points
                }
                pcl::PointXYZI point;
                point.x = p_data.x; point.y = p_data.y; point.z = p_data.z;
                point.intensity = p_data.intensity; // Absolute intensity
                cloud->points.push_back(point);
            }
        }

        if (cloud->points.empty()) {
             RCLCPP_ERROR(this->get_logger(), "Failed to create PCL cloud (0 valid points found). PLY not saved.");
            return false;
        }

        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;

        try {
            std::filesystem::path p(filepath);
            if (p.has_parent_path()) {
                if (!std::filesystem::exists(p.parent_path())) {
                     RCLCPP_INFO(this->get_logger(), "Creating directory: %s", p.parent_path().string().c_str());
                     std::filesystem::create_directories(p.parent_path());
                }
            }
            int result = pcl::io::savePLYFileBinary(filepath, *cloud);
            if (result == 0) {
                RCLCPP_INFO(this->get_logger(), "Successfully saved %zu points to %s", cloud->points.size(), filepath.c_str());
                return true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "PCL savePLYFile failed with error code %d for path %s", result, filepath.c_str());
                return false;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during PLY saving to %s: %s", filepath.c_str(), e.what());
            return false;
        }
    }

    void savePlyCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;
        std::filesystem::path dir_path(save_ply_directory_);
        std::filesystem::path base_path(save_ply_base_name_ + ".ply");
        std::string full_path = (dir_path / base_path).string();

        RCLCPP_INFO(this->get_logger(), "Manual Save PLY service triggered.");
        bool success = savePlyToFile(full_path);

        response->success = success;
        response->message = success ? "PLY file saved successfully to " + full_path
                                     : "Failed to save PLY file to " + full_path;
    }

    void onShutdownCallback() {
         RCLCPP_INFO(this->get_logger(), "Shutdown detected. Performing final save...");
         std::string timestamp = getTimestampString();
         std::filesystem::path dir_path(save_ply_directory_);
         std::filesystem::path file_path(save_ply_base_name_ + "_" + timestamp + ".ply");
         std::string full_path = (dir_path / file_path).string();
         bool success = savePlyToFile(full_path);
         if (success) {
             RCLCPP_INFO(this->get_logger(), "Shutdown save successful.");
         } else {
             RCLCPP_ERROR(this->get_logger(), "Shutdown save failed.");
         }
    }

    // --- Helper functions ---

    void publishEmptyCloud(){
        sensor_msgs::msg::PointCloud2 empty_msg;
        setupPointCloud2Msg(empty_msg, 0);
        accumulated_pc_pub_->publish(empty_msg);
    }

    void setupPointCloud2Msg(sensor_msgs::msg::PointCloud2& msg, size_t n_points){
        msg.header.stamp = this->now();
        msg.header.frame_id = map_frame_;
        msg.height = 1;
        msg.width = n_points;
        msg.point_step = 16;
        msg.row_step = msg.point_step * msg.width;
        msg.is_dense = true;
        msg.is_bigendian = false;
        msg.fields.resize(4);
        msg.fields[0].name = "x"; msg.fields[0].offset = 0; msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[0].count = 1;
        msg.fields[1].name = "y"; msg.fields[1].offset = 4; msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[1].count = 1;
        msg.fields[2].name = "z"; msg.fields[2].offset = 8; msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[2].count = 1;
        msg.fields[3].name = "intensity"; msg.fields[3].offset = 12; msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[3].count = 1;
        msg.data.resize(msg.row_step * msg.height);
        if (n_points == 0) { msg.data.clear(); }
    }

    void publishCentroid(double x, double y, double z, const rclcpp::Time& stamp){
        auto source_msg = std::make_unique<geometry_msgs::msg::PointStamped>();
        source_msg->header.stamp = stamp;
        source_msg->header.frame_id = map_frame_;
        source_msg->point.x = x; source_msg->point.y = y; source_msg->point.z = z;
        source_pub_->publish(std::move(source_msg));
    }

    // --- Member Variables ---
    std::string radiation_topic_, lidar_topic_, output_topic_, source_estimate_topic_;
    std::string sensor_frame_, map_frame_;
    double attenuation_mu_, min_cpm_, ray_max_distance_, min_intensity_publish_;
    double voxel_resolution_, publish_rate_;
    std::chrono::duration<double> tf_timeout_;
    bool save_on_shutdown_;
    std::string save_ply_directory_;
    std::string save_ply_base_name_;

    rclcpp::Clock::SharedPtr clock_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr accumulated_pc_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr source_pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rad_sub_separate_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_separate_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_ply_service_;
    // No need to store the shutdown handler if rclcpp::on_shutdown returns void

    double latest_cpm_ = 0.0;
    double global_max_absolute_intensity_ = 0.0;

    std::mutex voxel_map_mutex_;
    std::unordered_map<VoxelIndex, PointData, VoxelIndexHash> voxel_map_;

}; // End class RadiationPointcloudNode

// --- Main Function ---
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<RadiationPointcloudNode>(options);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown(); // Triggers registered on_shutdown callbacks
    return 0;
}