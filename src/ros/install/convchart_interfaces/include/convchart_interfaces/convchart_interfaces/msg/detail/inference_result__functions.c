// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from convchart_interfaces:msg/InferenceResult.idl
// generated code does not contain a copyright notice
#include "convchart_interfaces/msg/detail/inference_result__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `pose`
// Member `pose_alt`
#include "geometry_msgs/msg/detail/pose_with_covariance__functions.h"
// Member `reason`
#include "rosidl_runtime_c/string_functions.h"

bool
convchart_interfaces__msg__InferenceResult__init(convchart_interfaces__msg__InferenceResult * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    convchart_interfaces__msg__InferenceResult__fini(msg);
    return false;
  }
  // pose
  if (!geometry_msgs__msg__PoseWithCovariance__init(&msg->pose)) {
    convchart_interfaces__msg__InferenceResult__fini(msg);
    return false;
  }
  // rms
  // num_used
  // reason
  if (!rosidl_runtime_c__String__init(&msg->reason)) {
    convchart_interfaces__msg__InferenceResult__fini(msg);
    return false;
  }
  // ambiguous
  // pose_alt
  if (!geometry_msgs__msg__PoseWithCovariance__init(&msg->pose_alt)) {
    convchart_interfaces__msg__InferenceResult__fini(msg);
    return false;
  }
  // rms_alt
  // num_used_alt
  return true;
}

void
convchart_interfaces__msg__InferenceResult__fini(convchart_interfaces__msg__InferenceResult * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // pose
  geometry_msgs__msg__PoseWithCovariance__fini(&msg->pose);
  // rms
  // num_used
  // reason
  rosidl_runtime_c__String__fini(&msg->reason);
  // ambiguous
  // pose_alt
  geometry_msgs__msg__PoseWithCovariance__fini(&msg->pose_alt);
  // rms_alt
  // num_used_alt
}

bool
convchart_interfaces__msg__InferenceResult__are_equal(const convchart_interfaces__msg__InferenceResult * lhs, const convchart_interfaces__msg__InferenceResult * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__PoseWithCovariance__are_equal(
      &(lhs->pose), &(rhs->pose)))
  {
    return false;
  }
  // rms
  if (lhs->rms != rhs->rms) {
    return false;
  }
  // num_used
  if (lhs->num_used != rhs->num_used) {
    return false;
  }
  // reason
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->reason), &(rhs->reason)))
  {
    return false;
  }
  // ambiguous
  if (lhs->ambiguous != rhs->ambiguous) {
    return false;
  }
  // pose_alt
  if (!geometry_msgs__msg__PoseWithCovariance__are_equal(
      &(lhs->pose_alt), &(rhs->pose_alt)))
  {
    return false;
  }
  // rms_alt
  if (lhs->rms_alt != rhs->rms_alt) {
    return false;
  }
  // num_used_alt
  if (lhs->num_used_alt != rhs->num_used_alt) {
    return false;
  }
  return true;
}

bool
convchart_interfaces__msg__InferenceResult__copy(
  const convchart_interfaces__msg__InferenceResult * input,
  convchart_interfaces__msg__InferenceResult * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__PoseWithCovariance__copy(
      &(input->pose), &(output->pose)))
  {
    return false;
  }
  // rms
  output->rms = input->rms;
  // num_used
  output->num_used = input->num_used;
  // reason
  if (!rosidl_runtime_c__String__copy(
      &(input->reason), &(output->reason)))
  {
    return false;
  }
  // ambiguous
  output->ambiguous = input->ambiguous;
  // pose_alt
  if (!geometry_msgs__msg__PoseWithCovariance__copy(
      &(input->pose_alt), &(output->pose_alt)))
  {
    return false;
  }
  // rms_alt
  output->rms_alt = input->rms_alt;
  // num_used_alt
  output->num_used_alt = input->num_used_alt;
  return true;
}

convchart_interfaces__msg__InferenceResult *
convchart_interfaces__msg__InferenceResult__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  convchart_interfaces__msg__InferenceResult * msg = (convchart_interfaces__msg__InferenceResult *)allocator.allocate(sizeof(convchart_interfaces__msg__InferenceResult), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(convchart_interfaces__msg__InferenceResult));
  bool success = convchart_interfaces__msg__InferenceResult__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
convchart_interfaces__msg__InferenceResult__destroy(convchart_interfaces__msg__InferenceResult * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    convchart_interfaces__msg__InferenceResult__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
convchart_interfaces__msg__InferenceResult__Sequence__init(convchart_interfaces__msg__InferenceResult__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  convchart_interfaces__msg__InferenceResult * data = NULL;

  if (size) {
    if (size > SIZE_MAX / sizeof(convchart_interfaces__msg__InferenceResult)) {
      return false;
    }
    data = (convchart_interfaces__msg__InferenceResult *)allocator.zero_allocate(size, sizeof(convchart_interfaces__msg__InferenceResult), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = convchart_interfaces__msg__InferenceResult__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        convchart_interfaces__msg__InferenceResult__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
convchart_interfaces__msg__InferenceResult__Sequence__fini(convchart_interfaces__msg__InferenceResult__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      convchart_interfaces__msg__InferenceResult__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

convchart_interfaces__msg__InferenceResult__Sequence *
convchart_interfaces__msg__InferenceResult__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  convchart_interfaces__msg__InferenceResult__Sequence * array = (convchart_interfaces__msg__InferenceResult__Sequence *)allocator.allocate(sizeof(convchart_interfaces__msg__InferenceResult__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = convchart_interfaces__msg__InferenceResult__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
convchart_interfaces__msg__InferenceResult__Sequence__destroy(convchart_interfaces__msg__InferenceResult__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    convchart_interfaces__msg__InferenceResult__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
convchart_interfaces__msg__InferenceResult__Sequence__are_equal(const convchart_interfaces__msg__InferenceResult__Sequence * lhs, const convchart_interfaces__msg__InferenceResult__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!convchart_interfaces__msg__InferenceResult__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
convchart_interfaces__msg__InferenceResult__Sequence__copy(
  const convchart_interfaces__msg__InferenceResult__Sequence * input,
  convchart_interfaces__msg__InferenceResult__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    if (input->size > SIZE_MAX / sizeof(convchart_interfaces__msg__InferenceResult)) {
      return false;
    }
    const size_t allocation_size =
      input->size * sizeof(convchart_interfaces__msg__InferenceResult);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    convchart_interfaces__msg__InferenceResult * data =
      (convchart_interfaces__msg__InferenceResult *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!convchart_interfaces__msg__InferenceResult__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          convchart_interfaces__msg__InferenceResult__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!convchart_interfaces__msg__InferenceResult__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
