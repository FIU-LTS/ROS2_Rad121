#include "CUSB_RAD121.h"
#include <iostream>
#include <cmath>
#include <libftdi1/ftdi.h>

CUSB_RAD121::CUSB_RAD121()
{
    ftHandle = ftdi_new();
}

CUSB_RAD121::~CUSB_RAD121()
{
    Close();
    ftdi_free(ftHandle);
}

bool CUSB_RAD121::Open()
{
    if (ftdi_usb_open(ftHandle, 0x0403, 0x6001) < 0)
    {
        std::cerr << "Failed to open FTDI device" << std::endl;
        return false;
    }

    ftdi_set_baudrate(ftHandle, 921600);
    ftdi_set_line_property(ftHandle, BITS_8, STOP_BIT_1, NONE);
    ftdi_usb_purge_buffers(ftHandle);
    std::cerr << "FTDI device opened successfully" << std::endl;

    return true;
}

bool CUSB_RAD121::Close()
{
    return ftdi_usb_close(ftHandle) == 0;
}

int CUSB_RAD121::ReadData(unsigned char* buffer, int size)
{
    return ftdi_read_data(ftHandle, buffer, size);
}

void CUSB_RAD121::ConfigureFromNode(std::shared_ptr<rclcpp::Node> node)
{
    dead_time_ = node->declare_parameter("usb_rad121.dead_time", 0.00015);
    sensitivity_ = node->declare_parameter("usb_rad121.sensitivity", 450.0);
}

double CUSB_RAD121::Calculate_CompensatedCPM(double cpm)
{
    double cps = cpm / 60.0;
    double correction = 1.0 - cps * dead_time_;
    return (correction > 0.0) ? cpm / correction : 0.0;
}

double CUSB_RAD121::Calculate_mRhr(double cpm)
{
    return cpm / sensitivity_;
}
