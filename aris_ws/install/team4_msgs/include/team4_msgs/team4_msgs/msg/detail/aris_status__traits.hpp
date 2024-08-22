// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from team4_msgs:msg/ArisStatus.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__TRAITS_HPP_
#define TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "team4_msgs/msg/detail/aris_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace team4_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ArisStatus & msg,
  std::ostream & out)
{
  out << "{";
  // member: aris_status
  {
    out << "aris_status: ";
    rosidl_generator_traits::value_to_yaml(msg.aris_status, out);
    out << ", ";
  }

  // member: seat_number
  {
    out << "seat_number: ";
    rosidl_generator_traits::value_to_yaml(msg.seat_number, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ArisStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: aris_status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "aris_status: ";
    rosidl_generator_traits::value_to_yaml(msg.aris_status, out);
    out << "\n";
  }

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

inline std::string to_yaml(const ArisStatus & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace team4_msgs

namespace rosidl_generator_traits
{

[[deprecated("use team4_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const team4_msgs::msg::ArisStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  team4_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use team4_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const team4_msgs::msg::ArisStatus & msg)
{
  return team4_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<team4_msgs::msg::ArisStatus>()
{
  return "team4_msgs::msg::ArisStatus";
}

template<>
inline const char * name<team4_msgs::msg::ArisStatus>()
{
  return "team4_msgs/msg/ArisStatus";
}

template<>
struct has_fixed_size<team4_msgs::msg::ArisStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<team4_msgs::msg::ArisStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<team4_msgs::msg::ArisStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__TRAITS_HPP_
