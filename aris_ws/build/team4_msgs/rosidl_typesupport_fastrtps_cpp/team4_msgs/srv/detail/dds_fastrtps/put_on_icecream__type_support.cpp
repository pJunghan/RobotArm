// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from team4_msgs:srv/PutOnIcecream.idl
// generated code does not contain a copyright notice
#include "team4_msgs/srv/detail/put_on_icecream__rosidl_typesupport_fastrtps_cpp.hpp"
#include "team4_msgs/srv/detail/put_on_icecream__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace team4_msgs
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
cdr_serialize(
  const team4_msgs::srv::PutOnIcecream_Request & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: seat_number
  cdr << ros_message.seat_number;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  team4_msgs::srv::PutOnIcecream_Request & ros_message)
{
  // Member: seat_number
  cdr >> ros_message.seat_number;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
get_serialized_size(
  const team4_msgs::srv::PutOnIcecream_Request & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: seat_number
  {
    size_t item_size = sizeof(ros_message.seat_number);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
max_serialized_size_PutOnIcecream_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: seat_number
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = team4_msgs::srv::PutOnIcecream_Request;
    is_plain =
      (
      offsetof(DataType, seat_number) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _PutOnIcecream_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const team4_msgs::srv::PutOnIcecream_Request *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _PutOnIcecream_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<team4_msgs::srv::PutOnIcecream_Request *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _PutOnIcecream_Request__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const team4_msgs::srv::PutOnIcecream_Request *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _PutOnIcecream_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_PutOnIcecream_Request(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _PutOnIcecream_Request__callbacks = {
  "team4_msgs::srv",
  "PutOnIcecream_Request",
  _PutOnIcecream_Request__cdr_serialize,
  _PutOnIcecream_Request__cdr_deserialize,
  _PutOnIcecream_Request__get_serialized_size,
  _PutOnIcecream_Request__max_serialized_size
};

static rosidl_message_type_support_t _PutOnIcecream_Request__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_PutOnIcecream_Request__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace team4_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_team4_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<team4_msgs::srv::PutOnIcecream_Request>()
{
  return &team4_msgs::srv::typesupport_fastrtps_cpp::_PutOnIcecream_Request__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, team4_msgs, srv, PutOnIcecream_Request)() {
  return &team4_msgs::srv::typesupport_fastrtps_cpp::_PutOnIcecream_Request__handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include <limits>
// already included above
// #include <stdexcept>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
// already included above
// #include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace team4_msgs
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
cdr_serialize(
  const team4_msgs::srv::PutOnIcecream_Response & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: is_okay
  cdr << (ros_message.is_okay ? true : false);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  team4_msgs::srv::PutOnIcecream_Response & ros_message)
{
  // Member: is_okay
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_okay = tmp ? true : false;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
get_serialized_size(
  const team4_msgs::srv::PutOnIcecream_Response & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: is_okay
  {
    size_t item_size = sizeof(ros_message.is_okay);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team4_msgs
max_serialized_size_PutOnIcecream_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: is_okay
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = team4_msgs::srv::PutOnIcecream_Response;
    is_plain =
      (
      offsetof(DataType, is_okay) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _PutOnIcecream_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const team4_msgs::srv::PutOnIcecream_Response *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _PutOnIcecream_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<team4_msgs::srv::PutOnIcecream_Response *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _PutOnIcecream_Response__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const team4_msgs::srv::PutOnIcecream_Response *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _PutOnIcecream_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_PutOnIcecream_Response(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _PutOnIcecream_Response__callbacks = {
  "team4_msgs::srv",
  "PutOnIcecream_Response",
  _PutOnIcecream_Response__cdr_serialize,
  _PutOnIcecream_Response__cdr_deserialize,
  _PutOnIcecream_Response__get_serialized_size,
  _PutOnIcecream_Response__max_serialized_size
};

static rosidl_message_type_support_t _PutOnIcecream_Response__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_PutOnIcecream_Response__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace team4_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_team4_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<team4_msgs::srv::PutOnIcecream_Response>()
{
  return &team4_msgs::srv::typesupport_fastrtps_cpp::_PutOnIcecream_Response__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, team4_msgs, srv, PutOnIcecream_Response)() {
  return &team4_msgs::srv::typesupport_fastrtps_cpp::_PutOnIcecream_Response__handle;
}

#ifdef __cplusplus
}
#endif

#include "rmw/error_handling.h"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/service_type_support_decl.hpp"

namespace team4_msgs
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

static service_type_support_callbacks_t _PutOnIcecream__callbacks = {
  "team4_msgs::srv",
  "PutOnIcecream",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, team4_msgs, srv, PutOnIcecream_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, team4_msgs, srv, PutOnIcecream_Response)(),
};

static rosidl_service_type_support_t _PutOnIcecream__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_PutOnIcecream__callbacks,
  get_service_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace team4_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_team4_msgs
const rosidl_service_type_support_t *
get_service_type_support_handle<team4_msgs::srv::PutOnIcecream>()
{
  return &team4_msgs::srv::typesupport_fastrtps_cpp::_PutOnIcecream__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, team4_msgs, srv, PutOnIcecream)() {
  return &team4_msgs::srv::typesupport_fastrtps_cpp::_PutOnIcecream__handle;
}

#ifdef __cplusplus
}
#endif
