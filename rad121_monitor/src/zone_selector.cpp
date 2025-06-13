#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // For tf2::doTransform

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip> // For std::setprecision if needed for YAML output

#include "yaml-cpp/yaml.h" // For writing YAML

// Forward declaration if ZoneData is complex, or define inline
struct ZoneData {
    std::string label;
    double map_x;
    double map_y;
    double radius;
};

class ZoneSelectorNode : public rclcpp::Node {
public:
    ZoneSelectorNode() : Node("zone_selector_node"),
                        tf_buffer_(this->get_clock()),
                        tf_listener_(tf_buffer_),
                        zone_counter_(0)
    {
        declareParameters();
        getParameters();

        RCLCPP_INFO(this->get_logger(), "Zone Definer Node Initialized.");
        RCLCPP_INFO(this->get_logger(), "Listening for points on: %s", clicked_point_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Output YAML file will be: %s", output_yaml_file_.c_str());
        RCLCPP_INFO(this->get_logger(), "Default zone radius: %.2f m", default_zone_radius_);
        RCLCPP_INFO(this->get_logger(), "Target map frame: %s", map_frame_.c_str());

        point_subscriber_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            clicked_point_topic_, 10,
            std::bind(&ZoneSelectorNode::clickedPointCallback, this, std::placeholders::_1));

        save_zones_service_ = this->create_service<std_srvs::srv::Trigger>(
            "~/save_zones",
            std::bind(&ZoneSelectorNode::saveZonesCallback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Service '~/save_zones' available to write defined zones to YAML.");
    }

private:
    void declareParameters() {
        this->declare_parameter<std::string>("clicked_point_topic", "/clicked_point");
        this->declare_parameter<std::string>("output_yaml_file", "predefined_zones.yaml"); // Default to current dir
        this->declare_parameter<double>("default_zone_radius", 1.0); // meters
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<std::string>("zone_label_prefix", "Zone_");
    }

    void getParameters() {
        clicked_point_topic_ = this->get_parameter("clicked_point_topic").as_string();
        output_yaml_file_ = this->get_parameter("output_yaml_file").as_string();
        default_zone_radius_ = this->get_parameter("default_zone_radius").as_double();
        map_frame_ = this->get_parameter("map_frame").as_string();
        zone_label_prefix_ = this->get_parameter("zone_label_prefix").as_string();
    }

    void clickedPointCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
        geometry_msgs::msg::PointStamped point_in_map_frame;

        if (msg->header.frame_id.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received PointStamped with empty frame_id. Assuming it's in '%s'.", map_frame_.c_str());
            // If you want to strictly enforce frame_id, you could return here or throw an error.
            // For now, we'll try to use it as is if it matches map_frame_, otherwise transform.
            if (msg->header.frame_id != map_frame_ && !map_frame_.empty()) {
                 RCLCPP_ERROR(this->get_logger(), "Point has empty frame_id and target map_frame is '%s'. Cannot proceed without a source frame.", map_frame_.c_str());
                 return;
            }
            point_in_map_frame = *msg; // Assume it's in map_frame if frame_id is empty
            point_in_map_frame.header.frame_id = map_frame_; // Explicitly set it
        }
        else if (msg->header.frame_id != map_frame_) {
            try {
                point_in_map_frame = tf_buffer_.transform(*msg, map_frame_, tf2::durationFromSec(0.5));
            } catch (const tf2::TransformException &ex) {
                RCLCPP_ERROR(this->get_logger(), "Failed to transform point from '%s' to '%s': %s",
                             msg->header.frame_id.c_str(), map_frame_.c_str(), ex.what());
                return;
            }
        } else {
            point_in_map_frame = *msg; // Already in the target frame
        }

        ZoneData new_zone;
        new_zone.label = zone_label_prefix_ + generateZoneLetter(zone_counter_++);
        new_zone.map_x = point_in_map_frame.point.x;
        new_zone.map_y = point_in_map_frame.point.y;
        new_zone.radius = default_zone_radius_;

        defined_zones_.push_back(new_zone);

        RCLCPP_INFO(this->get_logger(), "Added zone: %s at (%.2f, %.2f) in '%s' frame with radius %.2f m. Total zones: %zu",
                    new_zone.label.c_str(), new_zone.map_x, new_zone.map_y, map_frame_.c_str(), new_zone.radius, defined_zones_.size());
    }

    std::string generateZoneLetter(int count) {
        if (count < 0) return "INVALID";
        std::string label = "";
        do {
            label = char('A' + (count % 26)) + label;
            count = count / 26 -1; // Next letter iteration
        } while (count >=0);
        return label;
    }


    void saveZonesCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                           std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request; // Unused

        if (defined_zones_.empty()) {
            RCLCPP_INFO(this->get_logger(), "No zones defined to save.");
            response->success = true; // Technically successful, as there's nothing to do
            response->message = "No zones defined to save.";
            return;
        }

        YAML::Node root_node;
        YAML::Node zones_list_node; // This will be a sequence

        for (const auto& zone : defined_zones_) {
            YAML::Node zone_node; // This will be a map
            zone_node["label"] = zone.label;
            zone_node["center_x"] = zone.map_x;
            zone_node["center_y"] = zone.map_y;
            zone_node["radius"] = zone.radius;
            zones_list_node.push_back(zone_node);
        }

        root_node["predefined_zones"] = zones_list_node;

        try {
            std::ofstream fout(output_yaml_file_);
            if (!fout.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open YAML file for writing: %s", output_yaml_file_.c_str());
                response->success = false;
                response->message = "Failed to open YAML file for writing: " + output_yaml_file_;
                return;
            }
            fout << root_node; // Write the YAML structure to the file
            fout.close();

            if (fout.good()) {
                 RCLCPP_INFO(this->get_logger(), "Successfully saved %zu zones to %s", defined_zones_.size(), output_yaml_file_.c_str());
                 response->success = true;
                 response->message = "Successfully saved " + std::to_string(defined_zones_.size()) + " zones to " + output_yaml_file_;
            } else {
                 RCLCPP_ERROR(this->get_logger(), "Error occurred during/after writing to YAML file: %s", output_yaml_file_.c_str());
                 response->success = false;
                 response->message = "Error occurred during/after writing to YAML file: " + output_yaml_file_;
            }

        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "YAML Exception during saving: %s", e.what());
            response->success = false;
            response->message = "YAML Exception: " + std::string(e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Standard Exception during saving: %s", e.what());
            response->success = false;
            response->message = "Standard Exception: " + std::string(e.what());
        }
    }

    // Member variables
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr point_subscriber_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_zones_service_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::string clicked_point_topic_;
    std::string output_yaml_file_;
    double default_zone_radius_;
    std::string map_frame_;
    std::string zone_label_prefix_;

    std::vector<ZoneData> defined_zones_;
    int zone_counter_; // For generating unique labels like Zone_A, Zone_B
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ZoneSelectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
