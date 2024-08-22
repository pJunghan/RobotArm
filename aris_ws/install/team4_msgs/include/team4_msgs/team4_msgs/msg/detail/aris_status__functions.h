// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from team4_msgs:msg/ArisStatus.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__FUNCTIONS_H_
#define TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "team4_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "team4_msgs/msg/detail/aris_status__struct.h"

/// Initialize msg/ArisStatus message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * team4_msgs__msg__ArisStatus
 * )) before or use
 * team4_msgs__msg__ArisStatus__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
bool
team4_msgs__msg__ArisStatus__init(team4_msgs__msg__ArisStatus * msg);

/// Finalize msg/ArisStatus message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
void
team4_msgs__msg__ArisStatus__fini(team4_msgs__msg__ArisStatus * msg);

/// Create msg/ArisStatus message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * team4_msgs__msg__ArisStatus__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
team4_msgs__msg__ArisStatus *
team4_msgs__msg__ArisStatus__create();

/// Destroy msg/ArisStatus message.
/**
 * It calls
 * team4_msgs__msg__ArisStatus__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
void
team4_msgs__msg__ArisStatus__destroy(team4_msgs__msg__ArisStatus * msg);

/// Check for msg/ArisStatus message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
bool
team4_msgs__msg__ArisStatus__are_equal(const team4_msgs__msg__ArisStatus * lhs, const team4_msgs__msg__ArisStatus * rhs);

/// Copy a msg/ArisStatus message.
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
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
bool
team4_msgs__msg__ArisStatus__copy(
  const team4_msgs__msg__ArisStatus * input,
  team4_msgs__msg__ArisStatus * output);

/// Initialize array of msg/ArisStatus messages.
/**
 * It allocates the memory for the number of elements and calls
 * team4_msgs__msg__ArisStatus__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
bool
team4_msgs__msg__ArisStatus__Sequence__init(team4_msgs__msg__ArisStatus__Sequence * array, size_t size);

/// Finalize array of msg/ArisStatus messages.
/**
 * It calls
 * team4_msgs__msg__ArisStatus__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
void
team4_msgs__msg__ArisStatus__Sequence__fini(team4_msgs__msg__ArisStatus__Sequence * array);

/// Create array of msg/ArisStatus messages.
/**
 * It allocates the memory for the array and calls
 * team4_msgs__msg__ArisStatus__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
team4_msgs__msg__ArisStatus__Sequence *
team4_msgs__msg__ArisStatus__Sequence__create(size_t size);

/// Destroy array of msg/ArisStatus messages.
/**
 * It calls
 * team4_msgs__msg__ArisStatus__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
void
team4_msgs__msg__ArisStatus__Sequence__destroy(team4_msgs__msg__ArisStatus__Sequence * array);

/// Check for msg/ArisStatus message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
bool
team4_msgs__msg__ArisStatus__Sequence__are_equal(const team4_msgs__msg__ArisStatus__Sequence * lhs, const team4_msgs__msg__ArisStatus__Sequence * rhs);

/// Copy an array of msg/ArisStatus messages.
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
ROSIDL_GENERATOR_C_PUBLIC_team4_msgs
bool
team4_msgs__msg__ArisStatus__Sequence__copy(
  const team4_msgs__msg__ArisStatus__Sequence * input,
  team4_msgs__msg__ArisStatus__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__FUNCTIONS_H_
