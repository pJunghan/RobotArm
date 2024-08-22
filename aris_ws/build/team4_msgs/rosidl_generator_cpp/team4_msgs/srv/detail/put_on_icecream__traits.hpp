// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from team4_msgs:srv/PutOnIcecream.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__TRAITS_HPP_
#define TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "team4_msgs/srv/detail/put_on_icecream__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace team4_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const PutOnIcecream_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: seat_number
  {
    out << "seat_number: ";
    rosidl_generator_traits::value_to_yaml(msg.seat_number, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PutOnIcecream_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: seat_number
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "seat_number: ";
    rosidl_generator_traits::value_to_yaml(msg.seat_number, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PutOnIcecream_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace team4_msgs

namespace rosidl_generator_traits
{

[[deprecated("use team4_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const team4_msgs::srv::PutOnIcecream_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  team4_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use team4_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const team4_msgs::srv::PutOnIcecream_Request & msg)
{
  return team4_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<team4_msgs::srv::PutOnIcecream_Request>()
{
  return "team4_msgs::srv::PutOnIcecream_Request";
}

template<>
inline const char * name<team4_msgs::srv::PutOnIcecream_Request>()
{
  return "team4_msgs/srv/PutOnIcecream_Request";
}

template<>
struct has_fixed_size<team4_msgs::srv::PutOnIcecream_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<team4_msgs::srv::PutOnIcecream_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<team4_msgs::srv::PutOnIcecream_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace team4_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const PutOnIcecream_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: is_okay
  {
    out << "is_okay: ";
    rosidl_generator_traits::value_to_yaml(msg.is_okay, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PutOnIcecream_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: is_okay
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_okay: ";
    rosidl_generator_traits::value_to_yaml(msg.is_okay, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PutOnIcecream_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace team4_msgs

namespace rosidl_generator_traits
{

[[deprecated("use team4_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const team4_msgs::srv::PutOnIcecream_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  team4_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use team4_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const team4_msgs::srv::PutOnIcecream_Response & msg)
{
  return team4_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<team4_msgs::srv::PutOnIcecream_Response>()
{
  return "team4_msgs::srv::PutOnIcecream_Response";
}

template<>
inline const char * name<team4_msgs::srv::PutOnIcecream_Response>()
{
  return "team4_msgs/srv/PutOnIcecream_Response";
}

template<>
struct has_fixed_size<team4_msgs::srv::PutOnIcecream_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<team4_msgs::srv::PutOnIcecream_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<team4_msgs::srv::PutOnIcecream_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<team4_msgs::srv::PutOnIcecream>()
{
  return "team4_msgs::srv::PutOnIcecream";
}

template<>
inline const char * name<team4_msgs::srv::PutOnIcecream>()
{
  return "team4_msgs/srv/PutOnIcecream";
}

template<>
struct has_fixed_size<team4_msgs::srv::PutOnIcecream>
  : std::integral_constant<
    bool,
    has_fixed_size<team4_msgs::srv::PutOnIcecream_Request>::value &&
    has_fixed_size<team4_msgs::srv::PutOnIcecream_Response>::value
  >
{
};

template<>
struct has_bounded_size<team4_msgs::srv::PutOnIcecream>
  : std::integral_constant<
    bool,
    has_bounded_size<team4_msgs::srv::PutOnIcecream_Request>::value &&
    has_bounded_size<team4_msgs::srv::PutOnIcecream_Response>::value
  >
{
};

template<>
struct is_service<team4_msgs::srv::PutOnIcecream>
  : std::true_type
{
};

template<>
struct is_service_request<team4_msgs::srv::PutOnIcecream_Request>
  : std::true_type
{
};

template<>
struct is_service_response<team4_msgs::srv::PutOnIcecream_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__TRAITS_HPP_
