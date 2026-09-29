// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from convchart_interfaces:msg/InferenceResult.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "convchart_interfaces/msg/inference_result.hpp"


#ifndef CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__TRAITS_HPP_
#define CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "convchart_interfaces/msg/detail/inference_result__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'pose'
// Member 'pose_alt'
#include "geometry_msgs/msg/detail/pose_with_covariance__traits.hpp"

namespace convchart_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const InferenceResult & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: pose
  {
    out << "pose: ";
    to_flow_style_yaml(msg.pose, out);
    out << ", ";
  }

  // member: rms
  {
    out << "rms: ";
    rosidl_generator_traits::value_to_yaml(msg.rms, out);
    out << ", ";
  }

  // member: num_used
  {
    out << "num_used: ";
    rosidl_generator_traits::value_to_yaml(msg.num_used, out);
    out << ", ";
  }

  // member: reason
  {
    out << "reason: ";
    rosidl_generator_traits::value_to_yaml(msg.reason, out);
    out << ", ";
  }

  // member: ambiguous
  {
    out << "ambiguous: ";
    rosidl_generator_traits::value_to_yaml(msg.ambiguous, out);
    out << ", ";
  }

  // member: pose_alt
  {
    out << "pose_alt: ";
    to_flow_style_yaml(msg.pose_alt, out);
    out << ", ";
  }

  // member: rms_alt
  {
    out << "rms_alt: ";
    rosidl_generator_traits::value_to_yaml(msg.rms_alt, out);
    out << ", ";
  }

  // member: num_used_alt
  {
    out << "num_used_alt: ";
    rosidl_generator_traits::value_to_yaml(msg.num_used_alt, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const InferenceResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pose:\n";
    to_block_style_yaml(msg.pose, out, indentation + 2);
  }

  // member: rms
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rms: ";
    rosidl_generator_traits::value_to_yaml(msg.rms, out);
    out << "\n";
  }

  // member: num_used
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_used: ";
    rosidl_generator_traits::value_to_yaml(msg.num_used, out);
    out << "\n";
  }

  // member: reason
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reason: ";
    rosidl_generator_traits::value_to_yaml(msg.reason, out);
    out << "\n";
  }

  // member: ambiguous
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ambiguous: ";
    rosidl_generator_traits::value_to_yaml(msg.ambiguous, out);
    out << "\n";
  }

  // member: pose_alt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pose_alt:\n";
    to_block_style_yaml(msg.pose_alt, out, indentation + 2);
  }

  // member: rms_alt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rms_alt: ";
    rosidl_generator_traits::value_to_yaml(msg.rms_alt, out);
    out << "\n";
  }

  // member: num_used_alt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_used_alt: ";
    rosidl_generator_traits::value_to_yaml(msg.num_used_alt, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const InferenceResult & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace convchart_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use convchart_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const convchart_interfaces::msg::InferenceResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  convchart_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use convchart_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const convchart_interfaces::msg::InferenceResult & msg)
{
  return convchart_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<convchart_interfaces::msg::InferenceResult>()
{
  return "convchart_interfaces::msg::InferenceResult";
}

template<>
inline const char * name<convchart_interfaces::msg::InferenceResult>()
{
  return "convchart_interfaces/msg/InferenceResult";
}

template<>
struct has_fixed_size<convchart_interfaces::msg::InferenceResult>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<convchart_interfaces::msg::InferenceResult>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<convchart_interfaces::msg::InferenceResult>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__TRAITS_HPP_
