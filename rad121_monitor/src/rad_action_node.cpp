#include "CUSB_RAD121.h"
#include <rclcpp/rclcpp.hpp>
#include "rclcpp_action/rclcpp_action.hpp"
#include "rad121_monitor_interfaces/action/rad.hpp"

class RadActionServer : public rclcpp::Node
{
public:
  using Rad = rad121_monitor_interfaces::action::Rad;
  using GoalHandleRad = rclcpp_action::ServerGoalHandle<Rad>;

    RadActionServer() : Node("rad_action_server")
    {
        rad121_ = std::make_shared<CUSB_RAD121>();
        if (!rad121_->Open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open FTDI device");
            rclcpp::shutdown();
        }
        using namespace std::placeholders;
        this->action_server_ = rclcpp_action::create_server<Rad>(
        this,
        "rad",
        std::bind(&RadActionServer::handle_goal, this, _1, _2),
        std::bind(&RadActionServer::handle_cancel, this, _1),
        std::bind(&RadActionServer::handle_accepted, this, _1));

        RCLCPP_INFO(this->get_logger(), "Rad Action Server is ready");
    }

    ~RadActionServer()
    {
        rad121_->Close();
    }

private:
rclcpp_action::Server<Rad>::SharedPtr action_server_;
std::shared_ptr<CUSB_RAD121> rad121_;
std::deque<std::chrono::steady_clock::time_point> countTimestamps;

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const Rad::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request with scan time %d", goal->time);
    (void)uuid;

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleRad> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleRad> goal_handle)
  {
    using namespace std::placeholders;
    // this needs to return quickly to avoid blocking the executor, so spin up a new thread
    std::thread{std::bind(&RadActionServer::execute, this, _1), goal_handle}.detach();
  }

  void execute(const std::shared_ptr<GoalHandleRad> goal_handle)
  {
    unsigned char buffer[64];
    rclcpp::Rate loop_rate(8);
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<Rad::Feedback>();
    auto & current_count = feedback->count;
    auto & time_remaining = feedback->sec_remain;
    auto result = std::make_shared<Rad::Result>();
    rclcpp::Time start_time = this->get_clock()->now();
    rclcpp::Time current_time = this->get_clock()->now();

    while((current_time.seconds() - start_time.seconds()) <= static_cast<double>(goal->time) && rclcpp::ok())
    {
        current_time = this->get_clock()->now();
        current_count += rad121_->ReadData(buffer, sizeof(buffer));
        feedback->count = current_count;
        RCLCPP_INFO(this->get_logger(), "Current count: %ld", current_count);

        // Check if there is an external cancel request
        if (goal_handle->is_canceling()) {
        result->count = current_count;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Scan canceled");
        return;
        }
        
        time_remaining = static_cast<double>(goal->time) - (current_time.seconds() - start_time.seconds());
        //RCLCPP_INFO(this->get_logger(), "Time remaining: %f", time_remaining);
        feedback->sec_remain = time_remaining;
        goal_handle->publish_feedback(feedback);
        loop_rate.sleep();
    }

    // Check if goal is done
    if (rclcpp::ok()) {
      result->count = current_count / goal->time;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Scan finished, final cps average: %ld", result->count);
    }
  } // function execute

}; // Class RadActionServer

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RadActionServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

