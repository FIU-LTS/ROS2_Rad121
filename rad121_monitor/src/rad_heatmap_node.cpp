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

class RadiationHeatmapNode : public rclcpp::Node
{
public:
  RadiationHeatmapNode()
  : Node("radiation_heatmap_node"),
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

    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", 10,
      std::bind(&RadiationHeatmapNode::mapCallback, this, std::placeholders::_1));

    rad_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/rad", 10,
      std::bind(&RadiationHeatmapNode::radCallback, this, std::placeholders::_1));

    rad_costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rad_heatmap", 10);

    RCLCPP_INFO(this->get_logger(), "Radiation Heatmap Node Initialized. Session folder: %s", session_folder_.c_str());

    rclcpp::on_shutdown([this]() { saveMap(); });
  }

private:
    struct RadiationSample {
        double x, y;
        double cpm;
        double yaw;
    };

    bool isVisible(int x0, int y0, int x1, int y1)
    {
        int dx = std::abs(x1 - x0);
        int dy = -std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        int width = base_map_.info.width;

        int max_range_cells = static_cast<int>(3.0 / base_map_.info.resolution);  // 2 meters max range
        int step_count = 0;

        while (true) {
        if (++step_count > max_range_cells) return false;
        int idx = y0 * width + x0;
        if (base_map_.data[idx] >= 50) return false;
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
      new_costmap.data.assign(new_costmap.info.width * new_costmap.info.height, 0);

      rad_costmap_ = new_costmap;
      rad_field_ = cv::Mat::zeros(new_costmap.info.height, new_costmap.info.width, CV_32FC1);

      RCLCPP_INFO(this->get_logger(), "Rad costmap and field initialized/resized to %ux%u",
                  new_costmap.info.width, new_costmap.info.height);
    }
  }

  void radCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    double cpm = msg->data[0];
    if (cpm < 400.0) {  // Lower threshold for visibility
      return;
    }

    geometry_msgs::msg::PoseStamped pose_in, pose_out;
    pose_in.header.frame_id = "base_link";
    pose_in.header.stamp = this->now();

    try {
      auto transform = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
      tf2::doTransform(pose_in, pose_out, transform);
    } catch (tf2::TransformException &ex) {
      return;
    }

    double rx = pose_out.pose.position.x;
    double ry = pose_out.pose.position.y;

    tf2::Quaternion q;
    tf2::fromMsg(pose_out.pose.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    {
    std::lock_guard<std::mutex> lock(map_mutex_);
    samples_.push_back({rx, ry, cpm});
    updateRadiationField();
    RCLCPP_INFO(this->get_logger(),
        "Mapped radiation reading: cpm=%.2f at (%.2f, %.2f).",
        cpm, rx, ry);
    }
  }

  void updateRadiationField()
  {
    rad_field_ = cv::Mat::zeros(base_map_.info.height, base_map_.info.width, CV_32FC1);
  
    double sigma_x = 2.0;  // Short axis (perpendicular to robot heading)
    double sigma_y = 2.0;  // Long axis (aligned with robot heading)
    double inv_2sigma2_x = 1.0 / (2.0 * sigma_x * sigma_x);
    double inv_2sigma2_y = 1.0 / (2.0 * sigma_y * sigma_y);
    double res = base_map_.info.resolution;
    double origin_x = base_map_.info.origin.position.x;
    double origin_y = base_map_.info.origin.position.y;
  
    for (int y = 0; y < rad_field_.rows; ++y) {
      for (int x = 0; x < rad_field_.cols; ++x) {
        double wx = origin_x + x * res + res / 2.0;
        double wy = origin_y + y * res + res / 2.0;
  
        double numerator = 0.0;
        double denominator = 0.0;
  
        for (const auto& sample : samples_) {
          if (sample.cpm < 100.0) continue;  // Suppress background
  
          int sx = static_cast<int>((sample.x - origin_x) / res);
          int sy = static_cast<int>((sample.y - origin_y) / res);
  
          if (sx < 0 || sx >= rad_field_.cols || sy < 0 || sy >= rad_field_.rows)
            continue;
          if (!isVisible(sx, sy, x, y))
            continue;
  
          double dx = wx - sample.x;
          double dy = wy - sample.y;
  
          // Rotate the difference vector into sample's robot frame
          double cos_yaw = std::cos(sample.yaw);
          double sin_yaw = std::sin(sample.yaw);
          double rx =  cos_yaw * dx + sin_yaw * dy;
          double ry = -sin_yaw * dx + cos_yaw * dy;
  
          // Elliptical Gaussian weight
          double weight = std::exp(-(rx * rx * inv_2sigma2_y + ry * ry * inv_2sigma2_x));
          numerator += weight * sample.cpm;
          denominator += weight;
        }
  
        float estimated = (denominator > 0.0) ? static_cast<float>(numerator / denominator) : 0.0f;
        rad_field_.at<float>(y, x) = estimated;
      }
    }
  
    // Optional: apply smoothing filter
    cv::GaussianBlur(rad_field_, rad_field_, cv::Size(5, 5), 1.5);
  
    // Autoscale for visualization
    float max_cpm = 0.0f;
    for (int y = 0; y < rad_field_.rows; ++y)
      for (int x = 0; x < rad_field_.cols; ++x)
        max_cpm = std::max(max_cpm, rad_field_.at<float>(y, x));
  
    float scale = (max_cpm > 0.0f) ? 100.0f / max_cpm : 0.0f;
  
    for (int y = 0; y < rad_field_.rows; ++y) {
      for (int x = 0; x < rad_field_.cols; ++x) {
        float value = rad_field_.at<float>(y, x) * scale;
        rad_costmap_.data[y * rad_field_.cols + x] =
          static_cast<int8_t>(std::clamp(value, 0.0f, 100.0f));
      }
    }
  
    rad_costmap_.header.stamp = this->now();
    rad_costmap_pub_->publish(rad_costmap_);
  }
  
  void saveMap()
  {
    std::string output_path = session_folder_ + "/heatmap.png";

    std::lock_guard<std::mutex> lock(map_mutex_);
    if (rad_field_.empty() || base_map_.data.empty()) {
      return;
    }

    int width = base_map_.info.width;
    int height = base_map_.info.height;

    cv::Mat map_img(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = y * width + x;
        int val = base_map_.data[idx];
        if (val == -1) map_img.at<uchar>(y, x) = 127;
        else if (val == 100) map_img.at<uchar>(y, x) = 0;
        else map_img.at<uchar>(y, x) = 255;
      }
    }

    cv::Mat map_bgr;
    cv::cvtColor(map_img, map_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat field_mask;
    rad_field_.convertTo(field_mask, CV_8UC1, 255.0 / 10000.0);
    cv::medianBlur(field_mask, field_mask, 3);  // remove salt-and-pepper noise
    cv::threshold(field_mask, field_mask, 10, 255, cv::THRESH_TOZERO);  // suppress near-zero noise
    cv::Mat field_color;
    cv::applyColorMap(field_mask, field_color, cv::COLORMAP_JET);

    cv::Mat blend;
    double alpha = 0.5;
    cv::addWeighted(map_bgr, 1.0 - alpha, field_color, alpha, 0.0, blend);

    cv::flip(blend, blend, 0);
    cv::imwrite(output_path, blend);
  }

  std::string session_folder_;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rad_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr rad_costmap_pub_;

  nav_msgs::msg::OccupancyGrid base_map_;
  nav_msgs::msg::OccupancyGrid rad_costmap_;
  cv::Mat rad_field_;
  std::vector<RadiationSample> samples_;
  std::mutex map_mutex_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RadiationHeatmapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}