#include "CUSB_RAD121.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <chrono>
#include <deque>

class Rad121MonitorNode : public rclcpp::Node
{
public:
Rad121MonitorNode()
    : Node("rad121_monitor_node"), decayRate(0.99), updateInterval(1)
{
    // Do NOT call shared_from_this() here
}

void init()
{
    rad121_ = std::make_shared<CUSB_RAD121>();
    rad121_->ConfigureFromNode(shared_from_this());

    if (!rad121_->Open())
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to open FTDI device");
        rclcpp::shutdown();
    }

    publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("rad", 10);
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000 / updateInterval),
        std::bind(&Rad121MonitorNode::MonitorClicks, this));
}

    ~Rad121MonitorNode()
    {
        rad121_->Close();
    }

private:
    void MonitorClicks()
    {
        unsigned char buffer[64];
        auto currentTime = std::chrono::steady_clock::now();

        int count = rad121_->ReadData(buffer, sizeof(buffer));
        if (count < 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Error reading data from FTDI device");
            rclcpp::shutdown();
        }

        if (count > 0)
        {
            for (int i = 0; i < count; ++i)
            {
                countTimestamps.push_back(currentTime);
            }
        }

        while (!countTimestamps.empty() &&
               std::chrono::duration_cast<std::chrono::seconds>(currentTime - countTimestamps.front()).count() >= 1)
        {
            countTimestamps.pop_front();
        }

        double cps = countTimestamps.size();
        double cpm = cps * 60.0;
        double cpm_comp = rad121_->Calculate_CompensatedCPM(cpm);
        double mR_hr = rad121_->Calculate_mRhr(cpm_comp);

        auto message = std_msgs::msg::Float64MultiArray();
        message.data.push_back(cpm_comp);
        message.data.push_back(mR_hr);
        publisher_->publish(message);

        RCLCPP_INFO(this->get_logger(), "Compensated CPM: %.2f | mR/hr: %.6f", cpm_comp, mR_hr);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    std::shared_ptr<CUSB_RAD121> rad121_;
    std::deque<std::chrono::steady_clock::time_point> countTimestamps;

    const double decayRate;
    const int updateInterval;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Rad121MonitorNode>();
    node->init();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

