cmake_minimum_required(VERSION 3.5)
project(rad121_monitor)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rad121_monitor_interfaces REQUIRED)
pkg_check_modules(LIBFTDI REQUIRED libftdi1)

include_directories(include ${LIBFTDI_INCLUDE_DIRS})

set(dependencies
rclcpp
rclcpp_action
rad121_monitor_interfaces
)

add_executable(rad_action_node src/rad_action_node.cpp src/CUSB_RAD121.cpp)

target_link_libraries(rad_action_node ${LIBFTDI_LIBRARIES})

ament_target_dependencies(rad_action_node ${dependencies})

install(TARGETS
rad_action_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()

