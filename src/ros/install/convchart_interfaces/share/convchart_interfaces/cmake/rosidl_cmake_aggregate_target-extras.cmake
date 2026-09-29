# generated from rosidl_cmake/cmake/rosidl_cmake_aggregate_target-extras.cmake.in

# Create a convenience aggregate target convchart_interfaces::convchart_interfaces
# that links all generated interface targets, so downstream packages can use
# a single modern CMake target name instead of ${convchart_interfaces_TARGETS}.
if(convchart_interfaces_TARGETS AND NOT TARGET convchart_interfaces::convchart_interfaces)
  add_library(convchart_interfaces::convchart_interfaces INTERFACE IMPORTED)
  set_target_properties(convchart_interfaces::convchart_interfaces PROPERTIES
    INTERFACE_LINK_LIBRARIES "${convchart_interfaces_TARGETS}")
endif()
