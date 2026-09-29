// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from convchart_interfaces:msg/InferenceResult.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "convchart_interfaces/msg/inference_result.hpp"


#ifndef CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__BUILDER_HPP_
#define CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "convchart_interfaces/msg/detail/inference_result__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace convchart_interfaces
{

namespace msg
{

namespace builder
{

class Init_InferenceResult_num_used_alt
{
public:
  explicit Init_InferenceResult_num_used_alt(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  ::convchart_interfaces::msg::InferenceResult num_used_alt(::convchart_interfaces::msg::InferenceResult::_num_used_alt_type arg)
  {
    msg_.num_used_alt = std::move(arg);
    return std::move(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_rms_alt
{
public:
  explicit Init_InferenceResult_rms_alt(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  Init_InferenceResult_num_used_alt rms_alt(::convchart_interfaces::msg::InferenceResult::_rms_alt_type arg)
  {
    msg_.rms_alt = std::move(arg);
    return Init_InferenceResult_num_used_alt(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_pose_alt
{
public:
  explicit Init_InferenceResult_pose_alt(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  Init_InferenceResult_rms_alt pose_alt(::convchart_interfaces::msg::InferenceResult::_pose_alt_type arg)
  {
    msg_.pose_alt = std::move(arg);
    return Init_InferenceResult_rms_alt(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_ambiguous
{
public:
  explicit Init_InferenceResult_ambiguous(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  Init_InferenceResult_pose_alt ambiguous(::convchart_interfaces::msg::InferenceResult::_ambiguous_type arg)
  {
    msg_.ambiguous = std::move(arg);
    return Init_InferenceResult_pose_alt(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_reason
{
public:
  explicit Init_InferenceResult_reason(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  Init_InferenceResult_ambiguous reason(::convchart_interfaces::msg::InferenceResult::_reason_type arg)
  {
    msg_.reason = std::move(arg);
    return Init_InferenceResult_ambiguous(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_num_used
{
public:
  explicit Init_InferenceResult_num_used(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  Init_InferenceResult_reason num_used(::convchart_interfaces::msg::InferenceResult::_num_used_type arg)
  {
    msg_.num_used = std::move(arg);
    return Init_InferenceResult_reason(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_rms
{
public:
  explicit Init_InferenceResult_rms(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  Init_InferenceResult_num_used rms(::convchart_interfaces::msg::InferenceResult::_rms_type arg)
  {
    msg_.rms = std::move(arg);
    return Init_InferenceResult_num_used(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_pose
{
public:
  explicit Init_InferenceResult_pose(::convchart_interfaces::msg::InferenceResult & msg)
  : msg_(msg)
  {}
  Init_InferenceResult_rms pose(::convchart_interfaces::msg::InferenceResult::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_InferenceResult_rms(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

class Init_InferenceResult_header
{
public:
  Init_InferenceResult_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_InferenceResult_pose header(::convchart_interfaces::msg::InferenceResult::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_InferenceResult_pose(msg_);
  }

private:
  ::convchart_interfaces::msg::InferenceResult msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::convchart_interfaces::msg::InferenceResult>()
{
  return convchart_interfaces::msg::builder::Init_InferenceResult_header();
}

}  // namespace convchart_interfaces

#endif  // CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__BUILDER_HPP_
