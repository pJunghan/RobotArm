// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from team4_msgs:msg/StoragyStatus.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "team4_msgs/msg/detail/storagy_status__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace team4_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void StoragyStatus_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) team4_msgs::msg::StoragyStatus(_init);
}

void StoragyStatus_fini_function(void * message_memory)
{
  auto typed_message = static_cast<team4_msgs::msg::StoragyStatus *>(message_memory);
  typed_message->~StoragyStatus();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember StoragyStatus_message_member_array[2] = {
  {
    "storagy_status",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team4_msgs::msg::StoragyStatus, storagy_status),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "seat_number",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team4_msgs::msg::StoragyStatus, seat_number),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers StoragyStatus_message_members = {
  "team4_msgs::msg",  // message namespace
  "StoragyStatus",  // message name
  2,  // number of fields
  sizeof(team4_msgs::msg::StoragyStatus),
  StoragyStatus_message_member_array,  // message members
  StoragyStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  StoragyStatus_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t StoragyStatus_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &StoragyStatus_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace team4_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<team4_msgs::msg::StoragyStatus>()
{
  return &::team4_msgs::msg::rosidl_typesupport_introspection_cpp::StoragyStatus_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, team4_msgs, msg, StoragyStatus)() {
  return &::team4_msgs::msg::rosidl_typesupport_introspection_cpp::StoragyStatus_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
