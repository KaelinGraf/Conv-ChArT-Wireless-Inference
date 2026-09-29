// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from convchart_interfaces:msg/InferenceResult.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "convchart_interfaces/msg/detail/inference_result__rosidl_typesupport_introspection_c.h"
#include "convchart_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "convchart_interfaces/msg/detail/inference_result__functions.h"
#include "convchart_interfaces/msg/detail/inference_result__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `pose`
// Member `pose_alt`
#include "geometry_msgs/msg/pose_with_covariance.h"
// Member `pose`
// Member `pose_alt`
#include "geometry_msgs/msg/detail/pose_with_covariance__rosidl_typesupport_introspection_c.h"
// Member `reason`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  convchart_interfaces__msg__InferenceResult__init(message_memory);
}

void convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_fini_function(void * message_memory)
{
  convchart_interfaces__msg__InferenceResult__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_member_array[9] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, pose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "rms",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, rms),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_used",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, num_used),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "reason",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, reason),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "ambiguous",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, ambiguous),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "pose_alt",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, pose_alt),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "rms_alt",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, rms_alt),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_used_alt",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(convchart_interfaces__msg__InferenceResult, num_used_alt),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_members = {
  "convchart_interfaces__msg",  // message namespace
  "InferenceResult",  // message name
  9,  // number of fields
  sizeof(convchart_interfaces__msg__InferenceResult),
  false,  // has_any_key_member_
  convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_member_array,  // message members
  convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_init_function,  // function to initialize message memory (memory has to be allocated)
  convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_type_support_handle = {
  0,
  &convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_members,
  get_message_typesupport_handle_function,
  &convchart_interfaces__msg__InferenceResult__get_type_hash,
  &convchart_interfaces__msg__InferenceResult__get_type_description,
  &convchart_interfaces__msg__InferenceResult__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_convchart_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, convchart_interfaces, msg, InferenceResult)() {
  convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PoseWithCovariance)();
  convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_member_array[6].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PoseWithCovariance)();
  if (!convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_type_support_handle.typesupport_identifier) {
    convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &convchart_interfaces__msg__InferenceResult__rosidl_typesupport_introspection_c__InferenceResult_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
