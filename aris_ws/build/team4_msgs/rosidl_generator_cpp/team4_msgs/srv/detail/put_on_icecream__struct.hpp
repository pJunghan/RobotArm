// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from team4_msgs:srv/PutOnIcecream.idl
// generated code does not contain a copyright notice

#ifndef TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__STRUCT_HPP_
#define TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__team4_msgs__srv__PutOnIcecream_Request __attribute__((deprecated))
#else
# define DEPRECATED__team4_msgs__srv__PutOnIcecream_Request __declspec(deprecated)
#endif

namespace team4_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PutOnIcecream_Request_
{
  using Type = PutOnIcecream_Request_<ContainerAllocator>;

  explicit PutOnIcecream_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->seat_number = 0l;
    }
  }

  explicit PutOnIcecream_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->seat_number = 0l;
    }
  }

  // field types and members
  using _seat_number_type =
    int32_t;
  _seat_number_type seat_number;

  // setters for named parameter idiom
  Type & set__seat_number(
    const int32_t & _arg)
  {
    this->seat_number = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team4_msgs__srv__PutOnIcecream_Request
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team4_msgs__srv__PutOnIcecream_Request
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PutOnIcecream_Request_ & other) const
  {
    if (this->seat_number != other.seat_number) {
      return false;
    }
    return true;
  }
  bool operator!=(const PutOnIcecream_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PutOnIcecream_Request_

// alias to use template instance with default allocator
using PutOnIcecream_Request =
  team4_msgs::srv::PutOnIcecream_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace team4_msgs


#ifndef _WIN32
# define DEPRECATED__team4_msgs__srv__PutOnIcecream_Response __attribute__((deprecated))
#else
# define DEPRECATED__team4_msgs__srv__PutOnIcecream_Response __declspec(deprecated)
#endif

namespace team4_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PutOnIcecream_Response_
{
  using Type = PutOnIcecream_Response_<ContainerAllocator>;

  explicit PutOnIcecream_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->is_okay = false;
    }
  }

  explicit PutOnIcecream_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->is_okay = false;
    }
  }

  // field types and members
  using _is_okay_type =
    bool;
  _is_okay_type is_okay;

  // setters for named parameter idiom
  Type & set__is_okay(
    const bool & _arg)
  {
    this->is_okay = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team4_msgs__srv__PutOnIcecream_Response
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team4_msgs__srv__PutOnIcecream_Response
    std::shared_ptr<team4_msgs::srv::PutOnIcecream_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PutOnIcecream_Response_ & other) const
  {
    if (this->is_okay != other.is_okay) {
      return false;
    }
    return true;
  }
  bool operator!=(const PutOnIcecream_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PutOnIcecream_Response_

// alias to use template instance with default allocator
using PutOnIcecream_Response =
  team4_msgs::srv::PutOnIcecream_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace team4_msgs

namespace team4_msgs
{

namespace srv
{

struct PutOnIcecream
{
  using Request = team4_msgs::srv::PutOnIcecream_Request;
  using Response = team4_msgs::srv::PutOnIcecream_Response;
};

}  // namespace srv

}  // namespace team4_msgs

#endif  // TEAM4_MSGS__SRV__DETAIL__PUT_ON_ICECREAM__STRUCT_HPP_
