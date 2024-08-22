// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from team4_msgs:msg/ArisStatus.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "team4_msgs/msg/detail/aris_status__rosidl_typesupport_introspection_c.h"
#include "team4_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "team4_msgs/msg/detail/aris_status__functions.h"
#include "team4_msgs/msg/detail/aris_status__struct.h"


// Include directives for member types
// Member `aris_status`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  team4_msgs__msg__ArisStatus__init(message_memory);
}

void team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_fini_function(void * message_memory)
{
  team4_msgs__msg__ArisStatus__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_member_array[2] = {
  {
    "aris_status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team4_msgs__msg__ArisStatus, aris_status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "seat_number",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team4_msgs__msg__ArisStatus, seat_number),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_members = {
  "team4_msgs__msg",  // message namespace
  "ArisStatus",  // message name
  2,  // number of fields
  sizeof(team4_msgs__msg__ArisStatus),
  team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_member_array,  // message members
  team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_type_support_handle = {
  0,
  &team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_team4_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, msg, ArisStatus)() {
  if (!team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_type_support_handle.typesupport_identifier) {
    team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &team4_msgs__msg__ArisStatus__rosidl_typesupport_introspection_c__ArisStatus_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
