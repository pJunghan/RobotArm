// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team4_msgs:srv/IsOkayIcecream.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__SRV__DETAIL__IS_OKAY_ICECREAM__BUILDER_HPP_
#define TEAM4_MSGS__SRV__DETAIL__IS_OKAY_ICECREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team4_msgs/srv/detail/is_okay_icecream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team4_msgs
{

namespace srv
{

namespace builder
{

class Init_IsOkayIcecream_Request_is_okay
{
public:
  Init_IsOkayIcecream_Request_is_okay()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::team4_msgs::srv::IsOkayIcecream_Request is_okay(::team4_msgs::srv::IsOkayIcecream_Request::_is_okay_type arg)
  {
    msg_.is_okay = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team4_msgs::srv::IsOkayIcecream_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::team4_msgs::srv::IsOkayIcecream_Request>()
{
  return team4_msgs::srv::builder::Init_IsOkayIcecream_Request_is_okay();
}

}  // namespace team4_msgs


namespace team4_msgs
{

namespace srv
{

namespace builder
{

class Init_IsOkayIcecream_Response_seat_number
{
public:
  Init_IsOkayIcecream_Response_seat_number()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::team4_msgs::srv::IsOkayIcecream_Response seat_number(::team4_msgs::srv::IsOkayIcecream_Response::_seat_number_type arg)
  {
    msg_.seat_number = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team4_msgs::srv::IsOkayIcecream_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::team4_msgs::srv::IsOkayIcecream_Response>()
{
  return team4_msgs::srv::builder::Init_IsOkayIcecream_Response_seat_number();
}

}  // namespace team4_msgs

#endif  // TEAM4_MSGS__SRV__DETAIL__IS_OKAY_ICECREAM__BUILDER_HPP_
