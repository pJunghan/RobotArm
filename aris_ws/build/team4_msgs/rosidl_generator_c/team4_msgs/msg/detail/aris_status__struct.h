// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team4_msgs:msg/ArisStatus.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__STRUCT_H_
#define TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'aris_status'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/ArisStatus in the package team4_msgs.
typedef struct team4_msgs__msg__ArisStatus
{
  rosidl_runtime_c__String aris_status;
  int32_t seat_number;
} team4_msgs__msg__ArisStatus;

// Struct for a sequence of team4_msgs__msg__ArisStatus.
typedef struct team4_msgs__msg__ArisStatus__Sequence
{
  team4_msgs__msg__ArisStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team4_msgs__msg__ArisStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__STRUCT_H_
