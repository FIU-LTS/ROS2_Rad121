#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <vector>
#include <mutex>
#include <fstream>
#include <opencv2/opencv.hpp>  // For saving as a PGM image
#include <sys/stat.h>  // For mkdir
#include <sys/types.h> // For mode_t
#include <filesystem> // Needed for pwd

class RadiationCostmapNode : public rclcpp::Node
{
public:
  RadiationCostmapNode()
  : Node("radiation_costmap_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", 10,
      std::bind(&RadiationCostmapNode::mapCallback, this, std::placeholders::_1));

    rad_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/rad", 10,
      std::bind(&RadiationCostmapNode::radCallback, this, std::placeholders::_1));

    rad_costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rad_costmap", 10);

    RCLCPP_INFO(this->get_logger(), "Radiation Costmap Node Initialized.");

    // Register shutdown callback to save the map
    rclcpp::on_shutdown([this]() { saveMap(); });
  }

private:
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    base_map_ = *msg;

    if (rad_costmap_.data.empty()) {
      rad_costmap_ = *msg;
      rad_costmap_.data.assign(rad_costmap_.data.size(), 0);
    }
  }

  void radCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 1) {
      RCLCPP_WARN(this->get_logger(), "Radiation message missing data.");
      return;
    }

    double cps = msg->data[0];

    if (cps < 1.0) {
      RCLCPP_INFO(this->get_logger(), "CPS %.2f is below threshold. No mapping.", cps);
      return;
    }

    geometry_msgs::msg::PoseStamped pose_in, pose_out;
    pose_in.header.frame_id = "base_link";
    pose_in.header.stamp = this->now();

    try {
      auto transform = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
      tf2::doTransform(pose_in, pose_out, transform);
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "TF transform error: %s", ex.what());
      return;
    }

    double rx = pose_out.pose.position.x;
    double ry = pose_out.pose.position.y;

    int map_index = -1;
    unsigned int width, height;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      width = rad_costmap_.info.width;
      height = rad_costmap_.info.height;
      double resolution = rad_costmap_.info.resolution;
      double origin_x = rad_costmap_.info.origin.position.x;
      double origin_y = rad_costmap_.info.origin.position.y;

      int cell_x = static_cast<int>((rx - origin_x) / resolution);
      int cell_y = static_cast<int>((ry - origin_y) / resolution);

      if (cell_x >= 0 && cell_x < static_cast<int>(width) &&
          cell_y >= 0 && cell_y < static_cast<int>(height)) {
        map_index = cell_y * width + cell_x;
      }
    }

    if (map_index < 0) {
      RCLCPP_WARN(this->get_logger(), "Pose out of map bounds. Skipping overlay.");
      return;
    }

    double max_cps = 10.0;
    double ratio = std::min(cps / max_cps, 1.0);
    int scaled_value = static_cast<int>(ratio * 100.0);

    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      rad_costmap_.data[map_index] = std::max(static_cast<int>(rad_costmap_.data[map_index]), scaled_value);
      rad_costmap_.header.stamp = this->now();
      rad_costmap_pub_->publish(rad_costmap_);
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Mapped radiation CPS=%.2f at (%.2f, %.2f) -> cell %d (scaled=%d).",
      cps, rx, ry, map_index, scaled_value
    );
  }
  
  void saveMap()
    {
    std::string save_path = std::filesystem::current_path().string() + "/rad_costmap_output";
    std::string yaml_path = save_path + "/radiation_map.yaml";
    std::string map_pgm_path = save_path + "/map.pgm";
    std::string costmap_pgm_path = save_path + "/costmap.pgm";
    std::string combined_pgm_path = save_path + "/map_with_costmap.pgm";
    std::string combined_png_path = save_path + "/map_with_costmap.png";

    // Ensure the directory exists
    struct stat info;
    if (stat(save_path.c_str(), &info) != 0) {
        if (mkdir(save_path.c_str(), 0775) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create directory: %s", save_path.c_str());
            return;
        }
    }

    std::lock_guard<std::mutex> lock(map_mutex_);
    if (rad_costmap_.data.empty() || base_map_.data.empty()) {
        RCLCPP_WARN(this->get_logger(), "No map or costmap data to save.");
        return;
    }

    int width = base_map_.info.width;
    int height = base_map_.info.height;

    // Save YAML metadata
    std::ofstream yaml_file(yaml_path);
    if (!yaml_file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open YAML file for writing.");
        return;
    }
    yaml_file << "image: map_with_costmap.pgm\n";
    yaml_file << "resolution: " << base_map_.info.resolution << "\n";
    yaml_file << "origin: [" << base_map_.info.origin.position.x << ", "
              << base_map_.info.origin.position.y << ", 0.0]\n";
    yaml_file << "occupied_thresh: 0.65\n";
    yaml_file << "free_thresh: 0.196\n";
    yaml_file << "negate: 0\n";
    yaml_file.close();

    // Create OpenCV images
    cv::Mat map_image(height, width, CV_8UC1);
    cv::Mat costmap_image(height, width, CV_8UC1);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;

            // Process `/map`
            int map_value = base_map_.data[index];
            if (map_value == -1) {
                map_image.at<uchar>(height - y - 1, x) = 127;  // Unknown (gray)
            } else if (map_value == 100) {
                map_image.at<uchar>(height - y - 1, x) = 0;  // Occupied (black)
            } else {
                map_image.at<uchar>(height - y - 1, x) = 255;  // Free (white)
            }

            // Process `/rad_costmap`
            int cost_value = rad_costmap_.data[index];
            costmap_image.at<uchar>(height - y - 1, x) = static_cast<uchar>(cost_value * 2.55);  // Scale 0-100 → 0-255
        }
    }

    // Apply heatmap coloring to the radiation costmap
    cv::Mat costmap_colored;
    cv::applyColorMap(costmap_image, costmap_colored, cv::COLORMAP_JET);  // Apply heatmap colors

    // Convert `/map` to a 3-channel grayscale image
    cv::Mat map_colored;
    cv::cvtColor(map_image, map_colored, cv::COLOR_GRAY2BGR);

    // Ensure the costmap is actually applied
    cv::Mat blended_image = map_colored.clone();

    // TODO: Convert blended image to grayscale for `.pgm`, and make it work.
    cv::Mat blended_gray;
    cv::cvtColor(blended_image, blended_gray, cv::COLOR_BGR2GRAY);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;

            // If there is radiation at this cell, overlay it
            if (rad_costmap_.data[index] > 0) {
                double alpha = 0.5;  // 50% transparency
                blended_image.at<cv::Vec3b>(y, x) = 
                    (1.0 - alpha) * map_colored.at<cv::Vec3b>(y, x) + 
                    (alpha * costmap_colored.at<cv::Vec3b>(y, x));
            }
        }
    }

    // Save images
    cv::imwrite(map_pgm_path, map_image);
    cv::imwrite(costmap_pgm_path, costmap_image);
    cv::imwrite(combined_pgm_path, blended_gray);
    cv::imwrite(combined_png_path, blended_image);  // PNG for full-color view

    RCLCPP_INFO(this->get_logger(), "Maps saved: %s, %s, %s, and %s",
                map_pgm_path.c_str(), costmap_pgm_path.c_str(), combined_pgm_path.c_str(), combined_png_path.c_str());
    }


  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rad_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr rad_costmap_pub_;

  nav_msgs::msg::OccupancyGrid base_map_;
  nav_msgs::msg::OccupancyGrid rad_costmap_;
  std::mutex map_mutex_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RadiationCostmapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
