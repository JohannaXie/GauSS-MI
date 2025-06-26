cmake_minimum_required(VERSION 3.0.2)
project(gs_mapping)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)