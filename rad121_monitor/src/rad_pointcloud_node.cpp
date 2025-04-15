#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp" // For reading input cloud
#include "geometry_msgs/msg/pose_stamped.hpp"    // Used for intermediate poses if needed
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/point_stamped.hpp" // For transforming points
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // TF conversions and doTransform
#include "tf2/LinearMath/Quaternion.h"             // For tf2::Quaternion
#include "tf2/LinearMath/Vector3.h"                // For tf2::Vector3
#include "tf2/LinearMath/Transform.h"              // For tf2::Transform operations

// Includes for message_filters
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <vector>
#include <cmath>
#include <string>
#include <limits> // Required for std::numeric_limits
#include <map>    // For storing best hits per lidar point index
#include <set>    // For storing unique hit indices

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define a type alias for the synchronization policy
typedef message_filters::sync_policies::ApproximateTime<
    std_msgs::msg::Float64MultiArray,
    sensor_msgs::msg::PointCloud2>
    ApproxSyncPolicy;

class RadiationPointcloudNode : public rclcpp::Node
{
public:
    // Constructor now takes NodeOptions to properly handle use_sim_time parameter
    explicit RadiationPointcloudNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : Node("radiation_pointcloud_node", options), // Pass options to base Node constructor
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_),
          sync_(nullptr) // Initialize synchronizer pointer to null
    {
        RCLCPP_INFO(this->get_logger(), "Initializing RadiationPointCloudGenerator Node (LiDAR Raycasting)...");

        // --- Declare and Load Parameters ---
        // Use node pointer 'this->' for parameter operations
        this->declare_parameter<std::string>("radiation_topic", "/rad");
        this->declare_parameter<std::string>("lidar_topic", "/velodyne_points");
        this->declare_parameter<std::string>("output_topic", "/rad_pointcloud");
        this->declare_parameter<std::string>("sensor_frame", "base_link"); // Recommend specific sensor frame
        this->declare_parameter<std::string>("map_frame", "map");          // Target frame for output cloud
        this->declare_parameter<double>("attenuation_coefficient_cm", 0.00012);
        this->declare_parameter<double>("min_cpm_threshold", 10.0);
        this->declare_parameter<double>("ray_max_distance", 5.0);
        this->declare_parameter<int>("rays_per_reading", 100);
        this->declare_parameter<double>("min_intensity_publish", 1.0);
        this->declare_parameter<double>("max_angle_match_rad", 0.1);

        // Get parameters using 'this->'
        radiation_topic_ = this->get_parameter("radiation_topic").as_string();
        lidar_topic_ = this->get_parameter("lidar_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        sensor_frame_ = this->get_parameter("sensor_frame").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        attenuation_mu_ = this->get_parameter("attenuation_coefficient_cm").as_double() * 100.0; // cm^-1 to m^-1
        min_cpm_ = this->get_parameter("min_cpm_threshold").as_double();
        ray_max_distance_ = this->get_parameter("ray_max_distance").as_double();
        rays_per_reading_ = this->get_parameter("rays_per_reading").as_int();
        min_intensity_publish_ = this->get_parameter("min_intensity_publish").as_double();
        max_angle_match_rad_ = this->get_parameter("max_angle_match_rad").as_double();

        // Log parameters
        RCLCPP_INFO(this->get_logger(), "Parameters loaded:");
        RCLCPP_INFO(this->get_logger(), "  Radiation topic: %s", radiation_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  LiDAR topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Output topic: %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Sensor frame: %s", sensor_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Map frame: %s", map_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Max angle match: %.3f rad", max_angle_match_rad_);
        RCLCPP_INFO(this->get_logger(), "  Max distance: %.2f m", ray_max_distance_);
        RCLCPP_INFO(this->get_logger(), "  Attenuation mu: %.5f m^-1", attenuation_mu_);
        RCLCPP_INFO(this->get_logger(), "  Min Intensity Publish: %.2f", min_intensity_publish_);

        // --- Subscribers using message_filters ---
        // Use node pointer 'this' for subscription creation
        rad_sub_.subscribe(this, radiation_topic_);
        lidar_sub_.subscribe(this, lidar_topic_);

        // --- Synchronizer ---
        // Queue size of 10 for ApproximateTime policy
        sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(ApproxSyncPolicy(10), rad_sub_, lidar_sub_);
        sync_->registerCallback(std::bind(&RadiationPointcloudNode::synchronizedCallback, this, std::placeholders::_1, std::placeholders::_2));

        // --- Publisher ---
        auto pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, pointcloud_qos);

        // --- Generate Ray Directions ---
        generateRayDirections(); // Generate unit vectors in sensor frame

        RCLCPP_INFO(this->get_logger(), "Radiation LiDAR Raycaster Node Initialized.");
    }

private:
    // Synchronized callback processing radiation and lidar data
    // Complete synchronizedCallback function with iterator reconstruction
    void synchronizedCallback(
        const std_msgs::msg::Float64MultiArray::ConstSharedPtr &rad_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar_msg)
    {
        // 1. Extract CPM and check threshold
        if (rad_msg->data.empty())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Received empty Float64MultiArray on %s", radiation_topic_.c_str());
            return;
        }
        double cpm = rad_msg->data[0]; // Assuming CPM is the first element

        if (cpm < min_cpm_)
        {
            return; // Skip low readings silently
        }
        // RCLCPP_INFO(this->get_logger(), "Sync Callback: CPM=%.2f", cpm); // Debug

        // 2. Get Transforms (Work in LiDAR frame for comparisons)
        geometry_msgs::msg::TransformStamped T_lidar_sensor; // Sensor pose relative to lidar
        geometry_msgs::msg::TransformStamped T_map_lidar;    // Lidar pose relative to map (for final output)
        rclcpp::Time target_time = lidar_msg->header.stamp;  // Use lidar timestamp for sync
        std::string lidar_frame = lidar_msg->header.frame_id;

        if (lidar_frame.empty())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Received LiDAR cloud with empty frame_id on %s", lidar_topic_.c_str());
            return;
        }

        try
        {
            tf2::Duration timeout = tf2::durationFromSec(0.15); // Slightly longer timeout for sync
            T_lidar_sensor = tf_buffer_.lookupTransform(lidar_frame, sensor_frame_, target_time, timeout);
            T_map_lidar = tf_buffer_.lookupTransform(map_frame_, lidar_frame, target_time, timeout);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "TF lookup failed: %s. Needed transforms between '%s', '%s', '%s' at time %.3f",
                                 ex.what(), map_frame_.c_str(), lidar_frame.c_str(), sensor_frame_.c_str(),
                                 rclcpp::Time(target_time).seconds());
            return;
        }

        // Sensor position and orientation in lidar frame
        tf2::Vector3 P_sensor_lidar_tf2;
        tf2::fromMsg(T_lidar_sensor.transform.translation, P_sensor_lidar_tf2);
        tf2::Quaternion Q_lidar_sensor_tf2;
        tf2::fromMsg(T_lidar_sensor.transform.rotation, Q_lidar_sensor_tf2);

        // 3. Check LiDAR Cloud status
        if (lidar_msg->width * lidar_msg->height == 0)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Received empty LiDAR cloud on %s", lidar_topic_.c_str());
            return;
        }

        // 4. Raycasting Loop (Option A - Nearest Angular Match per Ray)
        // Store best hit: Map LiDAR point index -> {intensity} pair
        // Stores the highest intensity found for that lidar point index across all rays
        std::map<uint32_t, float> best_intensity_hits;

        for (const auto &ray_dir_sensor_geo : ray_directions_)
        {
            // Rotate ray direction from sensor frame to lidar frame
            tf2::Vector3 ray_dir_sensor_tf2(ray_dir_sensor_geo.x, ray_dir_sensor_geo.y, ray_dir_sensor_geo.z);
            tf2::Vector3 ray_dir_lidar_tf2 = tf2::quatRotate(Q_lidar_sensor_tf2, ray_dir_sensor_tf2);
            ray_dir_lidar_tf2.normalize();

            int best_match_idx_for_ray = -1;
            double min_angle_diff_for_ray = max_angle_match_rad_;
            double best_match_dist_for_ray = std::numeric_limits<double>::max();

            // ---> RECONSTRUCT iterators for each ray's pass through the LiDAR cloud <---
            sensor_msgs::PointCloud2ConstIterator<float> iter_inner_x(*lidar_msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_inner_y(*lidar_msg, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iter_inner_z(*lidar_msg, "z");
            uint32_t current_lidar_idx_inner = 0;

            // Iterate through LiDAR points to find best match *for this specific ray*
            for (; iter_inner_x != iter_inner_x.end();
                 ++iter_inner_x, ++iter_inner_y, ++iter_inner_z, ++current_lidar_idx_inner)
            {
                // Lidar point in lidar frame
                tf2::Vector3 P_lidar_lidar_tf2(*iter_inner_x, *iter_inner_y, *iter_inner_z);

                // Vector from sensor to lidar point (in lidar frame)
                tf2::Vector3 V_lidar_tf2 = P_lidar_lidar_tf2 - P_sensor_lidar_tf2;
                double dist = V_lidar_tf2.length();

                if (dist > ray_max_distance_ || dist < 1e-3)
                    continue;

                tf2::Vector3 V_lidar_norm_tf2 = V_lidar_tf2.normalized();

                // Calculate angular difference
                double dot_product = ray_dir_lidar_tf2.dot(V_lidar_norm_tf2);
                dot_product = std::max(-1.0, std::min(1.0, dot_product)); // Clamp
                double angle_diff = std::acos(dot_product);

                // Check if this point is the best angular match for this ray
                if (angle_diff < min_angle_diff_for_ray)
                {
                    min_angle_diff_for_ray = angle_diff;
                    best_match_idx_for_ray = current_lidar_idx_inner;
                    best_match_dist_for_ray = dist;
                }
            } // End iterating through lidar points for one ray

            // If we found a suitable point for this ray
            if (best_match_idx_for_ray != -1)
            {
                double intensity = cpm * std::exp(-attenuation_mu_ * best_match_dist_for_ray);

                if (intensity >= min_intensity_publish_)
                {
                    // Check if this lidar point was already hit, if so, keep higher intensity
                    auto it = best_intensity_hits.find(best_match_idx_for_ray);
                    if (it == best_intensity_hits.end() || intensity > it->second)
                    {
                        best_intensity_hits[best_match_idx_for_ray] = static_cast<float>(intensity);
                    }
                }
            }
        } // End iterating through rays

        // 5. Create Output PointCloud2 Message with hit points transformed to map frame
        if (best_intensity_hits.empty())
        {
            return; // Nothing to publish
        }

        sensor_msgs::msg::PointCloud2 output_cloud_msg;
        output_cloud_msg.header.stamp = target_time;
        output_cloud_msg.header.frame_id = map_frame_; // Output cloud is in map frame
        output_cloud_msg.height = 1;
        output_cloud_msg.width = best_intensity_hits.size(); // Number of unique points hit
        output_cloud_msg.is_dense = true;                    // Assume transforms work unless exception occurs below
        output_cloud_msg.is_bigendian = false;

        // Define PointFields manually
        output_cloud_msg.fields.resize(4);
        output_cloud_msg.fields[0].name = "x";
        output_cloud_msg.fields[0].offset = 0;
        output_cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        output_cloud_msg.fields[0].count = 1;
        output_cloud_msg.fields[1].name = "y";
        output_cloud_msg.fields[1].offset = 4;
        output_cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        output_cloud_msg.fields[1].count = 1;
        output_cloud_msg.fields[2].name = "z";
        output_cloud_msg.fields[2].offset = 8;
        output_cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        output_cloud_msg.fields[2].count = 1;
        output_cloud_msg.fields[3].name = "intensity";
        output_cloud_msg.fields[3].offset = 12;
        output_cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
        output_cloud_msg.fields[3].count = 1;

        output_cloud_msg.point_step = 16; // 4 fields * 4 bytes/float
        output_cloud_msg.row_step = output_cloud_msg.point_step * output_cloud_msg.width;
        output_cloud_msg.data.resize(output_cloud_msg.row_step * output_cloud_msg.height);

        // Create Output Iterators
        sensor_msgs::PointCloud2Iterator<float> iter_out_x(output_cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_out_y(output_cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_out_z(output_cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_out_intensity(output_cloud_msg, "intensity");

        // ---> RECONSTRUCT input iterators for final pass to get hit point coords <---
        uint32_t current_lidar_idx_output = 0;
        sensor_msgs::PointCloud2ConstIterator<float> iter_lidar_x_final(*lidar_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_lidar_y_final(*lidar_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_lidar_z_final(*lidar_msg, "z");
        size_t points_added = 0; // Track successful additions

        // Iterate through the *input* cloud again to get coordinates of hit points
        for (; iter_lidar_x_final != iter_lidar_x_final.end();
             ++iter_lidar_x_final, ++iter_lidar_y_final, ++iter_lidar_z_final, ++current_lidar_idx_output)
        {
            auto hit_it = best_intensity_hits.find(current_lidar_idx_output);
            if (hit_it != best_intensity_hits.end()) // If this point was hit
            {
                geometry_msgs::msg::PointStamped p_lidar;
                p_lidar.header.frame_id = lidar_frame;
                p_lidar.header.stamp = target_time; // Use consistent timestamp for transform
                p_lidar.point.x = *iter_lidar_x_final;
                p_lidar.point.y = *iter_lidar_y_final;
                p_lidar.point.z = *iter_lidar_z_final;

                geometry_msgs::msg::PointStamped p_map;
                try
                {
                    // Use the transform we looked up earlier (Map -> LiDAR)
                    tf2::doTransform(p_lidar, p_map, T_map_lidar);

                    // Populate output cloud fields - check iterator validity implicitly by points_added
                    if (points_added < output_cloud_msg.width)
                    {
                        *iter_out_x = static_cast<float>(p_map.point.x);
                        *iter_out_y = static_cast<float>(p_map.point.y);
                        *iter_out_z = static_cast<float>(p_map.point.z);
                        *iter_out_intensity = hit_it->second; // Stored intensity

                        // Increment output iterators ONLY when adding a point
                        ++iter_out_x;
                        ++iter_out_y;
                        ++iter_out_z;
                        ++iter_out_intensity;
                        points_added++;
                    }
                    else
                    {
                        RCLCPP_ERROR(this->get_logger(), "PointCloud2 buffer overrun detected!");
                    }
                }
                catch (const tf2::TransformException &ex)
                {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                         "Failed to transform hit point index %d from '%s' to '%s': %s. Skipping point.",
                                         current_lidar_idx_output, lidar_frame.c_str(), map_frame_.c_str(), ex.what());
                    // If a point fails, we simply don't add it. Mark cloud potentially not dense.
                    output_cloud_msg.is_dense = false;
                }
            }
        }

        // Final adjustment of width if points were skipped due to TF errors
        if (points_added != output_cloud_msg.width)
        {
            RCLCPP_WARN(this->get_logger(), "Output cloud width adjusted from %u to %zu due to transform errors.", output_cloud_msg.width, points_added);
            output_cloud_msg.width = points_added;
            // Recalculate row_step and resize data buffer to potentially save memory
            output_cloud_msg.row_step = output_cloud_msg.point_step * output_cloud_msg.width;
            output_cloud_msg.data.resize(output_cloud_msg.row_step * output_cloud_msg.height);
        }

        // 6. Publish
        if (output_cloud_msg.width > 0)
        {
            pc_pub_->publish(output_cloud_msg);
            // RCLCPP_INFO(this->get_logger(), "Published radiation hits cloud with %zu points.", output_cloud_msg.width);
        }
    } // End of synchronizedCallback
    // Generates directions for rays (simple spherical distribution)
    void generateRayDirections()
    {
        ray_directions_.clear();
        ray_directions_.reserve(rays_per_reading_);
        double phi = M_PI * (3.0 - std::sqrt(5.0)); // Golden angle increment

        for (int i = 0; i < rays_per_reading_; ++i)
        {
            double y = 1.0 - (i / static_cast<double>(rays_per_reading_ - 1)) * 2.0;
            double radius = std::sqrt(1.0 - y * y);
            double theta = phi * i;
            geometry_msgs::msg::Vector3 dir;
            dir.x = std::cos(theta) * radius;
            dir.y = y;
            dir.z = std::sin(theta) * radius;
            // No need to normalize here if logic below handles it, but good practice:
            // double norm = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
            // if (norm > 1e-6) { dir.x /= norm; dir.y /= norm; dir.z /= norm;}
            ray_directions_.push_back(dir);
        }
        RCLCPP_INFO(this->get_logger(), "Generated %zu ray directions.", ray_directions_.size());
    }

    // Parameters (Member variables)
    std::string radiation_topic_;
    std::string lidar_topic_;
    std::string output_topic_;
    std::string sensor_frame_;
    std::string map_frame_;
    double attenuation_mu_; // m^-1
    double min_cpm_;
    double ray_max_distance_;
    int rays_per_reading_;
    double min_intensity_publish_;
    double max_angle_match_rad_;

    // ROS interfaces
    message_filters::Subscriber<std_msgs::msg::Float64MultiArray> rad_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_; // Use pointer

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Pre-calculated ray directions (unit vectors in sensor frame)
    std::vector<geometry_msgs::msg::Vector3> ray_directions_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);    // Handles ROS args
    rclcpp::NodeOptions options; // Create default options object
    auto node = std::make_shared<RadiationPointcloudNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}