cmake_minimum_required(VERSION 3.5)
project(rad121_monitor)

# Default to C++17 (ROS 2 recommends this)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBFTDI REQUIRED libftdi1)

# Include directories
include_directories(include ${LIBFTDI_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Radiation sensor node
add_executable(rad121_monitor_node src/rad121_monitor_node.cpp src/CUSB_RAD121.cpp)
target_link_libraries(rad121_monitor_node ${LIBFTDI_LIBRARIES})
ament_target_dependencies(rad121_monitor_node rclcpp std_msgs)

# Radiation costmap node (NEW)
add_executable(rad_costmap_node src/rad_costmap_node.cpp)
target_link_libraries(rad_costmap_node ${OpenCV_LIBS})
ament_target_dependencies(rad_costmap_node rclcpp std_msgs nav_msgs tf2 tf2_ros tf2_geometry_msgs OpenCV)

# Install targets
install(TARGETS
  rad121_monitor_node
  rad_costmap_node
  DESTINATION lib/${PROJECT_NAME})

# Install headers and launch files
install(DIRECTORY include/
  DESTINATION include/)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
