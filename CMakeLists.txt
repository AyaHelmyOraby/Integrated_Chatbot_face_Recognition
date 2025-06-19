cmake_minimum_required(VERSION 3.0.2)
project(integrated_chatbot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package()

install(DIRECTORY launch scripts config models data
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS scripts/integrated_chatbot_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
