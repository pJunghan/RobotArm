// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team4_msgs:srv/PutOnIcecream.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__STRUCT_H_
#define TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/PutOnIcecream in the package team4_msgs.
typedef struct team4_msgs__srv__PutOnIcecream_Request
{
  /// request
  int32_t seat_number;
} team4_msgs__srv__PutOnIcecream_Request;

// Struct for a sequence of team4_msgs__srv__PutOnIcecream_Request.
typedef struct team4_msgs__srv__PutOnIcecream_Request__Sequence
{
  team4_msgs__srv__PutOnIcecream_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team4_msgs__srv__PutOnIcecream_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/PutOnIcecream in the package team4_msgs.
typedef struct team4_msgs__srv__PutOnIcecream_Response
{
  /// response
  bool is_okay;
} team4_msgs__srv__PutOnIcecream_Response;

// Struct for a sequence of team4_msgs__srv__PutOnIcecream_Response.
typedef struct team4_msgs__srv__PutOnIcecream_Response__Sequence
{
  team4_msgs__srv__PutOnIcecream_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team4_msgs__srv__PutOnIcecream_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__STRUCT_H_
