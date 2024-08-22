// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team4_msgs:srv/PutOnIcecream.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__BUILDER_HPP_
#define TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team4_msgs/srv/detail/put_on_icecream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team4_msgs
{

namespace srv
{

namespace builder
{

class Init_PutOnIcecream_Request_seat_number
{
public:
  Init_PutOnIcecream_Request_seat_number()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::team4_msgs::srv::PutOnIcecream_Request seat_number(::team4_msgs::srv::PutOnIcecream_Request::_seat_number_type arg)
  {
    msg_.seat_number = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team4_msgs::srv::PutOnIcecream_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::team4_msgs::srv::PutOnIcecream_Request>()
{
  return team4_msgs::srv::builder::Init_PutOnIcecream_Request_seat_number();
}

}  // namespace team4_msgs


namespace team4_msgs
{

namespace srv
{

namespace builder
{

class Init_PutOnIcecream_Response_is_okay
{
public:
  Init_PutOnIcecream_Response_is_okay()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::team4_msgs::srv::PutOnIcecream_Response is_okay(::team4_msgs::srv::PutOnIcecream_Response::_is_okay_type arg)
  {
    msg_.is_okay = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team4_msgs::srv::PutOnIcecream_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::team4_msgs::srv::PutOnIcecream_Response>()
{
  return team4_msgs::srv::builder::Init_PutOnIcecream_Response_is_okay();
}

}  // namespace team4_msgs

#endif  // TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__BUILDER_HPP_
