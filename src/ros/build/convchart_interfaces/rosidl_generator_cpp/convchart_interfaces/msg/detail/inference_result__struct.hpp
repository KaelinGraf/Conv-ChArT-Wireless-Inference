// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from convchart_interfaces:msg/InferenceResult.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "convchart_interfaces/msg/inference_result.hpp"


#ifndef CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__STRUCT_HPP_
#define CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'pose'
// Member 'pose_alt'
#include "geometry_msgs/msg/detail/pose_with_covariance__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__convchart_interfaces__msg__InferenceResult __attribute__((deprecated))
#else
# define DEPRECATED__convchart_interfaces__msg__InferenceResult __declspec(deprecated)
#endif

namespace convchart_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct InferenceResult_
{
  using Type = InferenceResult_<ContainerAllocator>;

  explicit InferenceResult_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    pose(_init),
    pose_alt(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->rms = 0.0f;
      this->num_used = 0;
      this->reason = "";
      this->ambiguous = false;
      this->rms_alt = 0.0f;
      this->num_used_alt = 0;
    }
  }

  explicit InferenceResult_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    pose(_alloc, _init),
    reason(_alloc),
    pose_alt(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->rms = 0.0f;
      this->num_used = 0;
      this->reason = "";
      this->ambiguous = false;
      this->rms_alt = 0.0f;
      this->num_used_alt = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _pose_type =
    geometry_msgs::msg::PoseWithCovariance_<ContainerAllocator>;
  _pose_type pose;
  using _rms_type =
    float;
  _rms_type rms;
  using _num_used_type =
    uint8_t;
  _num_used_type num_used;
  using _reason_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _reason_type reason;
  using _ambiguous_type =
    bool;
  _ambiguous_type ambiguous;
  using _pose_alt_type =
    geometry_msgs::msg::PoseWithCovariance_<ContainerAllocator>;
  _pose_alt_type pose_alt;
  using _rms_alt_type =
    float;
  _rms_alt_type rms_alt;
  using _num_used_alt_type =
    uint8_t;
  _num_used_alt_type num_used_alt;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__pose(
    const geometry_msgs::msg::PoseWithCovariance_<ContainerAllocator> & _arg)
  {
    this->pose = _arg;
    return *this;
  }
  Type & set__rms(
    const float & _arg)
  {
    this->rms = _arg;
    return *this;
  }
  Type & set__num_used(
    const uint8_t & _arg)
  {
    this->num_used = _arg;
    return *this;
  }
  Type & set__reason(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->reason = _arg;
    return *this;
  }
  Type & set__ambiguous(
    const bool & _arg)
  {
    this->ambiguous = _arg;
    return *this;
  }
  Type & set__pose_alt(
    const geometry_msgs::msg::PoseWithCovariance_<ContainerAllocator> & _arg)
  {
    this->pose_alt = _arg;
    return *this;
  }
  Type & set__rms_alt(
    const float & _arg)
  {
    this->rms_alt = _arg;
    return *this;
  }
  Type & set__num_used_alt(
    const uint8_t & _arg)
  {
    this->num_used_alt = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    convchart_interfaces::msg::InferenceResult_<ContainerAllocator> *;
  using ConstRawPtr =
    const convchart_interfaces::msg::InferenceResult_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      convchart_interfaces::msg::InferenceResult_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      convchart_interfaces::msg::InferenceResult_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__convchart_interfaces__msg__InferenceResult
    std::shared_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__convchart_interfaces__msg__InferenceResult
    std::shared_ptr<convchart_interfaces::msg::InferenceResult_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const InferenceResult_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->pose != other.pose) {
      return false;
    }
    if (this->rms != other.rms) {
      return false;
    }
    if (this->num_used != other.num_used) {
      return false;
    }
    if (this->reason != other.reason) {
      return false;
    }
    if (this->ambiguous != other.ambiguous) {
      return false;
    }
    if (this->pose_alt != other.pose_alt) {
      return false;
    }
    if (this->rms_alt != other.rms_alt) {
      return false;
    }
    if (this->num_used_alt != other.num_used_alt) {
      return false;
    }
    return true;
  }
  bool operator!=(const InferenceResult_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct InferenceResult_

// alias to use template instance with default allocator
using InferenceResult =
  convchart_interfaces::msg::InferenceResult_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace convchart_interfaces

#endif  // CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__STRUCT_HPP_
