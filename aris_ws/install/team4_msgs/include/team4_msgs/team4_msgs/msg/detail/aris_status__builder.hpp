// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team4_msgs:msg/ArisStatus.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__BUILDER_HPP_
#define TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team4_msgs/msg/detail/aris_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team4_msgs
{

namespace msg
{

namespace builder
{

class Init_ArisStatus_seat_number
{
public:
  explicit Init_ArisStatus_seat_number(::team4_msgs::msg::ArisStatus & msg)
  : msg_(msg)
  {}
  ::team4_msgs::msg::ArisStatus seat_number(::team4_msgs::msg::ArisStatus::_seat_number_type arg)
  {
    msg_.seat_number = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team4_msgs::msg::ArisStatus msg_;
};

class Init_ArisStatus_aris_status
{
public:
  Init_ArisStatus_aris_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ArisStatus_seat_number aris_status(::team4_msgs::msg::ArisStatus::_aris_status_type arg)
  {
    msg_.aris_status = std::move(arg);
    return Init_ArisStatus_seat_number(msg_);
  }

private:
  ::team4_msgs::msg::ArisStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::team4_msgs::msg::ArisStatus>()
{
  return team4_msgs::msg::builder::Init_ArisStatus_aris_status();
}

}  // namespace team4_msgs

#endif  // TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__BUILDER_HPP_
