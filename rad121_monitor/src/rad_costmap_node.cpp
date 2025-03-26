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
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

class RadiationCostmapNode : public rclcpp::Node
{
public:
  RadiationCostmapNode()
  : Node("radiation_costmap_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    std::time_t now = std::time(nullptr);
    std::stringstream ss;
    ss << "rad_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    session_folder_ = ss.str();

    if (!std::filesystem::exists(session_folder_)) {
      std::filesystem::create_directory(session_folder_);
    }

    csv_file_ = session_folder_ + "/rad_data.csv";

    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", 10,
      std::bind(&RadiationCostmapNode::mapCallback, this, std::placeholders::_1));

    rad_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/rad", 10,
      std::bind(&RadiationCostmapNode::radCallback, this, std::placeholders::_1));

    rad_costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rad_costmap", 10);

    RCLCPP_INFO(this->get_logger(),
                "Radiation Costmap Node Initialized. Session folder: %s",
                session_folder_.c_str());

    rclcpp::on_shutdown([this]() { saveMap(); });
  }

private:
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(map_mutex_);

    bool map_changed = false;
    if (base_map_.info.width != msg->info.width ||
        base_map_.info.height != msg->info.height ||
        base_map_.info.resolution != msg->info.resolution ||
        base_map_.info.origin.position.x != msg->info.origin.position.x ||
        base_map_.info.origin.position.y != msg->info.origin.position.y) {
      map_changed = true;
    }

    base_map_ = *msg;

    if (rad_costmap_.data.empty() || map_changed) {
      nav_msgs::msg::OccupancyGrid new_costmap = *msg;
      new_costmap.info.origin = base_map_.info.origin;       // force costmap to align spatially
      new_costmap.info.resolution = base_map_.info.resolution;
      new_costmap.data.assign(new_costmap.info.width * new_costmap.info.height, 0);      
    
      if (!rad_costmap_.data.empty() && !map_changed) {
        rad_costmap_ = new_costmap;
        RCLCPP_INFO(this->get_logger(), "Rad costmap initialized.");
        return;
      }
    
      // Preserve old data
      double old_resolution = rad_costmap_.info.resolution;
      double old_origin_x = rad_costmap_.info.origin.position.x;
      double old_origin_y = rad_costmap_.info.origin.position.y;
      unsigned int old_width = rad_costmap_.info.width;
      unsigned int old_height = rad_costmap_.info.height;
    
      for (unsigned int y = 0; y < old_height; ++y) {
        for (unsigned int x = 0; x < old_width; ++x) {
          int old_idx = y * old_width + x;
          int old_value = rad_costmap_.data[old_idx];
          if (old_value == 0)
            continue;
    
          // Convert old cell to world coordinates
          double world_x = old_origin_x + x * old_resolution + old_resolution / 2.0;
          double world_y = old_origin_y + y * old_resolution + old_resolution / 2.0;
    
          // Convert to new map index
          int new_x = static_cast<int>((world_x - new_costmap.info.origin.position.x) / new_costmap.info.resolution);
          int new_y = static_cast<int>((world_y - new_costmap.info.origin.position.y) / new_costmap.info.resolution);
    
          if (new_x >= 0 && new_x < static_cast<int>(new_costmap.info.width) &&
              new_y >= 0 && new_y < static_cast<int>(new_costmap.info.height)) {
            int new_idx = new_y * new_costmap.info.width + new_x;
            int8_t old_val = static_cast<int8_t>(old_value);
            int8_t new_val = new_costmap.data[new_idx];
            new_costmap.data[new_idx] = std::max(new_val, old_val);            
          }
        }
      }
    
      rad_costmap_ = new_costmap;
      RCLCPP_INFO(this->get_logger(), "Rad costmap resized. New size: %ux%u",
                  rad_costmap_.info.width, rad_costmap_.info.height);
        if (base_map_.info.origin.position.x != rad_costmap_.info.origin.position.x ||
          base_map_.info.origin.position.y != rad_costmap_.info.origin.position.y ||
          base_map_.info.resolution != rad_costmap_.info.resolution) {
        RCLCPP_WARN(this->get_logger(), "WARNING: Base map and radiation costmap are not aligned (origin/resolution mismatch)!");
      }       
    }    
  }

  void radCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    double cpm = msg->data[0];
    double mrem_per_hr = msg->data[1];

    double rx = 0.0, ry = 0.0;
    bool valid_pose = false;
    int map_index_for_log = -1;

    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      geometry_msgs::msg::PoseStamped pose_in, pose_out;
      pose_in.header.frame_id = "base_link";
      pose_in.header.stamp = this->now();

      try {
        auto transform = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
        tf2::doTransform(pose_in, pose_out, transform);
        rx = pose_out.pose.position.x;
        ry = pose_out.pose.position.y;
        valid_pose = true;
      } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "TF transform error (logging anyway): %s", ex.what());
      }

      std::ofstream file(csv_file_, std::ios::app);
      if (file.is_open()) {
        file << cpm << "," << mrem_per_hr << "," << map_index_for_log << "," << rx << "," << ry << "\n";
        file.close();
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_file_.c_str());
      }
    }

    if (cpm < 1200.0) {
      RCLCPP_INFO(this->get_logger(), "cpm %.2f is below threshold. No mapping. (But CSV was logged)", cpm);
      return;
    }

    geometry_msgs::msg::PoseStamped pose_in, pose_out;
    pose_in.header.frame_id = "base_link";
    pose_in.header.stamp = this->now();

    try {
      auto transform = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
      tf2::doTransform(pose_in, pose_out, transform);
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "TF transform error for costmapping: %s", ex.what());
      return;
    }

    rx = pose_out.pose.position.x;
    ry = pose_out.pose.position.y;

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
      RCLCPP_WARN(this->get_logger(), "Pose out of map bounds. Skipping overlay. (But CSV was logged)");
      return;
    }

    double max_cpm = 10000.0;
    double ratio = std::min(cpm / max_cpm, 1.0);
    int scaled_value = static_cast<int>(ratio * 100.0);

    int dynamic_radius = std::clamp(static_cast<int>(cpm / 1000.0), 1, 10);

    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      int main_cell_y = map_index / width;
      int main_cell_x = map_index % width;

      for (int dy = -dynamic_radius; dy <= dynamic_radius; ++dy) {
        for (int dx = -dynamic_radius; dx <= dynamic_radius; ++dx) {
          int nx = main_cell_x + dx;
          int ny = main_cell_y + dy;

          if (nx >= 0 && nx < static_cast<int>(width) &&
              ny >= 0 && ny < static_cast<int>(height)) {
            int neighbor_idx = ny * width + nx;

            double distance = std::sqrt(dx*dx + dy*dy);
            double falloff = std::max(0.0, 1.0 - (distance / (dynamic_radius + 1)));

            int local_scaled_value = static_cast<int>(scaled_value * falloff);
            rad_costmap_.data[neighbor_idx] = std::min(
              rad_costmap_.data[neighbor_idx] + local_scaled_value,
              100
          );
          }
        }
      }

      rad_costmap_.header.stamp = this->now();
      rad_costmap_pub_->publish(rad_costmap_);
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Mapped radiation cpm=%.2f at (%.2f, %.2f) -> cell %d (radius=%d, scaled=%d).",
      cpm, rx, ry, map_index, dynamic_radius, scaled_value
    );
  }

  void saveMap()
  {
    std::string save_path = session_folder_;
    std::string yaml_path = save_path + "/radiation_map.yaml";
    std::string map_pgm_path = save_path + "/map.pgm";
    std::string costmap_pgm_path = save_path + "/costmap.pgm";
    std::string combined_pgm_path = save_path + "/map_with_costmap.pgm";
    std::string combined_png_path = save_path + "/map_with_costmap.png";
  
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
  
    // Build grayscale maps (without vertical flip here)
    cv::Mat map_image(height, width, CV_8UC1);
    cv::Mat costmap_image(height, width, CV_8UC1);
  
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int index = y * width + x;
  
        int map_value = base_map_.data[index];
        if (map_value == -1) {
          map_image.at<uchar>(y, x) = 127;
        } else if (map_value == 100) {
          map_image.at<uchar>(y, x) = 0;
        } else {
          map_image.at<uchar>(y, x) = 255;
        }
  
        int cost_value = rad_costmap_.data[index];
        costmap_image.at<uchar>(y, x) = static_cast<uchar>(cost_value * 2.55);
      }
    }
  
    // Apply colormap to costmap
    cv::Mat costmap_colored;
    cv::applyColorMap(costmap_image, costmap_colored, cv::COLORMAP_JET);
  
    // Convert grayscale map to 3-channel
    cv::Mat map_colored;
    cv::cvtColor(map_image, map_colored, cv::COLOR_GRAY2BGR);
  
    // Blend costmap over base map
    cv::Mat blended_image = map_colored.clone();
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int index = y * width + x;
        if (rad_costmap_.data[index] > 0) {
          double alpha = 0.5;
          blended_image.at<cv::Vec3b>(y, x) =
              (1.0 - alpha) * map_colored.at<cv::Vec3b>(y, x) +
               alpha        * costmap_colored.at<cv::Vec3b>(y, x);
        }
      }
    }
  
    // Flip images vertically to match ROS map coordinate convention
    cv::Mat flipped_map, flipped_costmap, flipped_blended, blended_gray;
    cv::flip(map_image, flipped_map, 0);
    cv::flip(costmap_image, flipped_costmap, 0);
    cv::flip(blended_image, flipped_blended, 0);
  
    // Convert flipped blend to grayscale (for PGM export)
    cv::cvtColor(flipped_blended, blended_gray, cv::COLOR_BGR2GRAY);
  
    // Save all images
    cv::imwrite(map_pgm_path,      flipped_map);
    cv::imwrite(costmap_pgm_path,  flipped_costmap);
    cv::imwrite(combined_pgm_path, blended_gray);
    cv::imwrite(combined_png_path, flipped_blended);

    // Create a compact colorbar legend
    int legend_height = 128;
    int legend_width = 40;
    cv::Mat legend(legend_height, 1, CV_8UC1);
    for (int i = 0; i < legend_height; ++i) {
      legend.at<uchar>(i, 0) = static_cast<uchar>(255 - (i * 255 / legend_height));  // inverse vertical gradient
    }
    cv::Mat legend_color;
    cv::applyColorMap(legend, legend_color, cv::COLORMAP_JET);

    // Resize to final dimensions
    cv::resize(legend_color, legend_color, cv::Size(legend_width, legend_height));

    // Add tick marks and labels
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.35;
    int thickness = 1;

    // Labels: top (max), middle, bottom (min)
    cv::putText(legend_color, "10000", cv::Point(2, 10), font, font_scale, cv::Scalar(255,255,255), thickness);
    cv::putText(legend_color, "5600",  cv::Point(5, legend_height / 2 + 5), font, font_scale, cv::Scalar(255,255,255), thickness);
    cv::putText(legend_color, "1200",  cv::Point(5, legend_height - 5), font, font_scale, cv::Scalar(255,255,255), thickness);

    // Add units label at the top (above legend)
    int label_height = 20;
    cv::Mat label(label_height, legend_width, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::putText(label, "CPM", cv::Point(3, label_height - 5), font, font_scale, cv::Scalar(255,255,255), thickness);

    // Stack label on top of legend
    cv::Mat full_legend;
    cv::vconcat(label, legend_color, full_legend);

    // Resize to match map height (optional: bottom align instead of stretching)
    cv::Mat legend_padded;
    int pad_height = flipped_blended.rows - full_legend.rows;
    cv::copyMakeBorder(full_legend, legend_padded, 0, pad_height, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    // Concatenate horizontally with the map
    cv::Mat combined_with_legend;
    cv::hconcat(flipped_blended, legend_padded, combined_with_legend);

    // Save the final image
    std::string legend_output_path = save_path + "/map_with_costmap_legend.png";
    cv::imwrite(legend_output_path, combined_with_legend);
    RCLCPP_INFO(this->get_logger(), "Map with legend saved: %s", legend_output_path.c_str());
  
    RCLCPP_INFO(this->get_logger(),
                "Maps saved in session folder: %s\n - %s\n - %s\n - %s\n - %s",
                save_path.c_str(),
                map_pgm_path.c_str(),
                costmap_pgm_path.c_str(),
                combined_pgm_path.c_str(),
                combined_png_path.c_str());
  }  

  std::string session_folder_;
  std::string csv_file_;

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
