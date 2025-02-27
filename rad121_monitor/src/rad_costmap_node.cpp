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
#include <filesystem>  // Needed for the session folder

class RadiationCostmapNode : public rclcpp::Node
{
public:
  RadiationCostmapNode()
  : Node("radiation_costmap_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // ------------------------- 1) Create a timestamped session folder -------------------------
    std::time_t now = std::time(nullptr);
    std::stringstream ss;
    ss << "rad_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    session_folder_ = ss.str();

    if (!std::filesystem::exists(session_folder_)) {
      std::filesystem::create_directory(session_folder_);
    }

    // Path to the CSV file (optional, if you want to log hits).
    csv_file_ = session_folder_ + "/rad_data.csv";

    // ------------------------- Original subscriptions & publishers -------------------------
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

    // Register shutdown callback to save the map
    rclcpp::on_shutdown([this]() { saveMap(); });
  }

private:
  // Session folder + CSV file path
  std::string session_folder_;
  std::string csv_file_;

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

    double cpm = msg->data[0];
    double mrem_per_hr = msg->data[1];

    if (cpm < 1200.0) {
      RCLCPP_INFO(this->get_logger(), "cpm %.2f is below threshold. No mapping.", cpm);
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

    double max_cpm = 10000.0;
    double ratio = std::min(cpm / max_cpm, 1.0);
    int scaled_value = static_cast<int>(ratio * 100.0);

    // ------------------------- 2) Scatter-like expansion around the single cell -------------------------
    int expansion_radius = 2.5; // set to 1, or larger if desired
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      for (int dy = -expansion_radius; dy <= expansion_radius; ++dy) {
        for (int dx = -expansion_radius; dx <= expansion_radius; ++dx) {
          int cell_y = map_index / width;
          int cell_x = map_index % width;

          int nx = cell_x + dx;
          int ny = cell_y + dy;

          if (nx >= 0 && nx < static_cast<int>(width) &&
              ny >= 0 && ny < static_cast<int>(height)) {
            int neighbor_idx = ny * width + nx;
            rad_costmap_.data[neighbor_idx] = std::max(
                static_cast<int>(rad_costmap_.data[neighbor_idx]),
                scaled_value
            );
          }
        }
      }

      rad_costmap_.header.stamp = this->now();
      rad_costmap_pub_->publish(rad_costmap_);
    }

    // Optional: Log the detection to the CSV (if you want).
    {
      std::ofstream file(csv_file_, std::ios::app);
      if (file.is_open()) {
        file << cpm << "," << mrem_per_hr << "," << map_index << ","
             << rx << "," << ry << "\n";
        file.close();
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open %s for writing.", csv_file_.c_str());
      }
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Mapped radiation cpm=%.2f at (%.2f, %.2f) -> cell %d (scaled=%d).",
      cpm, rx, ry, map_index, scaled_value
    );
  }
  
  void saveMap()
  {
    // ------------------------- 3) Save map images, but use the session folder -------------------------
    std::string save_path = session_folder_; // changed from current_path + "/rad_costmap_output"
    std::string yaml_path = save_path + "/radiation_map.yaml";
    std::string map_pgm_path = save_path + "/map.pgm";
    std::string costmap_pgm_path = save_path + "/costmap.pgm";
    std::string combined_pgm_path = save_path + "/map_with_costmap.pgm";
    std::string combined_png_path = save_path + "/map_with_costmap.png";

    // Ensure the directory exists (should already exist from constructor)
    // but we'll check again in case the user manually removed it
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

        // Fill map_image
        int map_value = base_map_.data[index];
        if (map_value == -1) {
          map_image.at<uchar>(height - y - 1, x) = 127;  // Unknown (gray)
        } else if (map_value == 100) {
          map_image.at<uchar>(height - y - 1, x) = 0;  // Occupied (black)
        } else {
          map_image.at<uchar>(height - y - 1, x) = 255; // Free (white)
        }

        // Fill costmap_image (0..100 -> 0..255)
        int cost_value = rad_costmap_.data[index];
        costmap_image.at<uchar>(height - y - 1, x) = static_cast<uchar>(cost_value * 2.55);
      }
    }

    // Heatmap coloring for the costmap
    cv::Mat costmap_colored;
    cv::applyColorMap(costmap_image, costmap_colored, cv::COLORMAP_JET);

    // Convert map_image to a 3-channel grayscale
    cv::Mat map_colored;
    cv::cvtColor(map_image, map_colored, cv::COLOR_GRAY2BGR);

    // Blend them
    cv::Mat blended_image = map_colored.clone();
    cv::Mat blended_gray;
    cv::cvtColor(blended_image, blended_gray, cv::COLOR_BGR2GRAY);

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

    // Save images into the session folder
    cv::imwrite(map_pgm_path,      map_image);
    cv::imwrite(costmap_pgm_path,  costmap_image);
    cv::imwrite(combined_pgm_path, blended_gray);
    cv::imwrite(combined_png_path, blended_image);

    RCLCPP_INFO(this->get_logger(),
                "Maps saved in session folder: %s\n - %s\n - %s\n - %s\n - %s",
                save_path.c_str(),
                map_pgm_path.c_str(),
                costmap_pgm_path.c_str(),
                combined_pgm_path.c_str(),
                combined_png_path.c_str());
  }

  // Original subscriber/publisher members
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rad_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr rad_costmap_pub_;

  // Original stored maps
  nav_msgs::msg::OccupancyGrid base_map_;
  nav_msgs::msg::OccupancyGrid rad_costmap_;
  std::mutex map_mutex_;

  // TF buffer & listener
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
