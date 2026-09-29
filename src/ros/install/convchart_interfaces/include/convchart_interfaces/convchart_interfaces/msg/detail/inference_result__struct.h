// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from convchart_interfaces:msg/InferenceResult.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "convchart_interfaces/msg/inference_result.h"


#ifndef CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__STRUCT_H_
#define CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'pose'
// Member 'pose_alt'
#include "geometry_msgs/msg/detail/pose_with_covariance__struct.h"
// Member 'reason'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/InferenceResult in the package convchart_interfaces.
typedef struct convchart_interfaces__msg__InferenceResult
{
  std_msgs__msg__Header header;
  geometry_msgs__msg__PoseWithCovariance pose;
  /// RMS of pose result
  float rms;
  /// num of corners
  uint8_t num_used;
  /// if non-empty, then the inference result is invalid
  rosidl_runtime_c__String reason;
  /// if ambiguous, compute suprise
  bool ambiguous;
  geometry_msgs__msg__PoseWithCovariance pose_alt;
  /// RMS of pose result
  float rms_alt;
  /// num of corners
  uint8_t num_used_alt;
} convchart_interfaces__msg__InferenceResult;

// Struct for a sequence of convchart_interfaces__msg__InferenceResult.
typedef struct convchart_interfaces__msg__InferenceResult__Sequence
{
  convchart_interfaces__msg__InferenceResult * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} convchart_interfaces__msg__InferenceResult__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__STRUCT_H_
