// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from team4_msgs:msg/ArisStatus.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__STRUCT_HPP_
#define TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__team4_msgs__msg__ArisStatus __attribute__((deprecated))
#else
# define DEPRECATED__team4_msgs__msg__ArisStatus __declspec(deprecated)
#endif

namespace team4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ArisStatus_
{
  using Type = ArisStatus_<ContainerAllocator>;

  explicit ArisStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->aris_status = "";
      this->seat_number = 0l;
    }
  }

  explicit ArisStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : aris_status(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->aris_status = "";
      this->seat_number = 0l;
    }
  }

  // field types and members
  using _aris_status_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _aris_status_type aris_status;
  using _seat_number_type =
    int32_t;
  _seat_number_type seat_number;

  // setters for named parameter idiom
  Type & set__aris_status(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->aris_status = _arg;
    return *this;
  }
  Type & set__seat_number(
    const int32_t & _arg)
  {
    this->seat_number = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team4_msgs::msg::ArisStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const team4_msgs::msg::ArisStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team4_msgs::msg::ArisStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team4_msgs::msg::ArisStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team4_msgs__msg__ArisStatus
    std::shared_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team4_msgs__msg__ArisStatus
    std::shared_ptr<team4_msgs::msg::ArisStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ArisStatus_ & other) const
  {
    if (this->aris_status != other.aris_status) {
      return false;
    }
    if (this->seat_number != other.seat_number) {
      return false;
    }
    return true;
  }
  bool operator!=(const ArisStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ArisStatus_

// alias to use template instance with default allocator
using ArisStatus =
  team4_msgs::msg::ArisStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace team4_msgs

#endif  // TEAM4_MSGS__MSG__DETAIL__ARIS_STATUS__STRUCT_HPP_
