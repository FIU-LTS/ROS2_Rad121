#ifndef CUSB_RAD121_H
#define CUSB_RAD121_H

#include <ftdi.h>
#include <chrono>
#include <deque>
#include <memory>
#include <rclcpp/rclcpp.hpp>

class CUSB_RAD121
{
public:
    CUSB_RAD121();
    ~CUSB_RAD121();

    bool Open();
    bool Close();
    int ReadData(unsigned char* buffer, int size);

    // New method for YAML param configuration
    void ConfigureFromNode(std::shared_ptr<rclcpp::Node> node);

    // Radiation calculations
    double Calculate_CompensatedCPM(double CountsPerMinute);
    double Calculate_mRhr(double CountsPerMinute);

private:
    struct ftdi_context* ftHandle;
    std::chrono::steady_clock::time_point LastCountsTime;

    // Parameters (configurable via YAML)
    double dead_time_ = 0.00015;    // seconds
    double sensitivity_ = 450.0;    // counts per mR/hr
};

#endif // CUSB_RAD121_H
