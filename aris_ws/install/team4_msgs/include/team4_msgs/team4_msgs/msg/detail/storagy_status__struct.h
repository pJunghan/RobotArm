// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team4_msgs:msg/StoragyStatus.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__MSG__DETAIL__STORAGY_STATUS__STRUCT_H_
#define TEAM4_MSGS__MSG__DETAIL__STORAGY_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'storagy_status'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/StoragyStatus in the package team4_msgs.
typedef struct team4_msgs__msg__StoragyStatus
{
  rosidl_runtime_c__String storagy_status;
  int32_t seat_number;
} team4_msgs__msg__StoragyStatus;

// Struct for a sequence of team4_msgs__msg__StoragyStatus.
typedef struct team4_msgs__msg__StoragyStatus__Sequence
{
  team4_msgs__msg__StoragyStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team4_msgs__msg__StoragyStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM4_MSGS__MSG__DETAIL__STORAGY_STATUS__STRUCT_H_
