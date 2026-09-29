// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from convchart_interfaces:msg/InferenceResult.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "convchart_interfaces/msg/inference_result.h"


#ifndef CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__FUNCTIONS_H_
#define CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/action_type_support_struct.h"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_runtime_c/type_description/type_description__struct.h"
#include "rosidl_runtime_c/type_description/type_source__struct.h"
#include "rosidl_runtime_c/type_hash.h"
#include "rosidl_runtime_c/visibility_control.h"
#include "convchart_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "convchart_interfaces/msg/detail/inference_result__struct.h"

/// Initialize msg/InferenceResult message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * convchart_interfaces__msg__InferenceResult
 * )) before or use
 * convchart_interfaces__msg__InferenceResult__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
bool
convchart_interfaces__msg__InferenceResult__init(convchart_interfaces__msg__InferenceResult * msg);

/// Finalize msg/InferenceResult message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
void
convchart_interfaces__msg__InferenceResult__fini(convchart_interfaces__msg__InferenceResult * msg);

/// Create msg/InferenceResult message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * convchart_interfaces__msg__InferenceResult__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
convchart_interfaces__msg__InferenceResult *
convchart_interfaces__msg__InferenceResult__create(void);

/// Destroy msg/InferenceResult message.
/**
 * It calls
 * convchart_interfaces__msg__InferenceResult__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
void
convchart_interfaces__msg__InferenceResult__destroy(convchart_interfaces__msg__InferenceResult * msg);

/// Check for msg/InferenceResult message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
bool
convchart_interfaces__msg__InferenceResult__are_equal(const convchart_interfaces__msg__InferenceResult * lhs, const convchart_interfaces__msg__InferenceResult * rhs);

/// Copy a msg/InferenceResult message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
bool
convchart_interfaces__msg__InferenceResult__copy(
  const convchart_interfaces__msg__InferenceResult * input,
  convchart_interfaces__msg__InferenceResult * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
const rosidl_type_hash_t *
convchart_interfaces__msg__InferenceResult__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
convchart_interfaces__msg__InferenceResult__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
const rosidl_runtime_c__type_description__TypeSource *
convchart_interfaces__msg__InferenceResult__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
convchart_interfaces__msg__InferenceResult__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of msg/InferenceResult messages.
/**
 * It allocates the memory for the number of elements and calls
 * convchart_interfaces__msg__InferenceResult__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
bool
convchart_interfaces__msg__InferenceResult__Sequence__init(convchart_interfaces__msg__InferenceResult__Sequence * array, size_t size);

/// Finalize array of msg/InferenceResult messages.
/**
 * It calls
 * convchart_interfaces__msg__InferenceResult__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
void
convchart_interfaces__msg__InferenceResult__Sequence__fini(convchart_interfaces__msg__InferenceResult__Sequence * array);

/// Create array of msg/InferenceResult messages.
/**
 * It allocates the memory for the array and calls
 * convchart_interfaces__msg__InferenceResult__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
convchart_interfaces__msg__InferenceResult__Sequence *
convchart_interfaces__msg__InferenceResult__Sequence__create(size_t size);

/// Destroy array of msg/InferenceResult messages.
/**
 * It calls
 * convchart_interfaces__msg__InferenceResult__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
void
convchart_interfaces__msg__InferenceResult__Sequence__destroy(convchart_interfaces__msg__InferenceResult__Sequence * array);

/// Check for msg/InferenceResult message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
bool
convchart_interfaces__msg__InferenceResult__Sequence__are_equal(const convchart_interfaces__msg__InferenceResult__Sequence * lhs, const convchart_interfaces__msg__InferenceResult__Sequence * rhs);

/// Copy an array of msg/InferenceResult messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_convchart_interfaces
bool
convchart_interfaces__msg__InferenceResult__Sequence__copy(
  const convchart_interfaces__msg__InferenceResult__Sequence * input,
  convchart_interfaces__msg__InferenceResult__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CONVCHART_INTERFACES__MSG__DETAIL__INFERENCE_RESULT__FUNCTIONS_H_
