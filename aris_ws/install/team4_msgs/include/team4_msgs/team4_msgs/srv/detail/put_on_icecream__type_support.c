// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from team4_msgs:srv/PutOnIcecream.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "team4_msgs/srv/detail/put_on_icecream__rosidl_typesupport_introspection_c.h"
#include "team4_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "team4_msgs/srv/detail/put_on_icecream__functions.h"
#include "team4_msgs/srv/detail/put_on_icecream__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  team4_msgs__srv__PutOnIcecream_Request__init(message_memory);
}

void team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_fini_function(void * message_memory)
{
  team4_msgs__srv__PutOnIcecream_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_member_array[1] = {
  {
    "seat_number",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team4_msgs__srv__PutOnIcecream_Request, seat_number),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_members = {
  "team4_msgs__srv",  // message namespace
  "PutOnIcecream_Request",  // message name
  1,  // number of fields
  sizeof(team4_msgs__srv__PutOnIcecream_Request),
  team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_member_array,  // message members
  team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_type_support_handle = {
  0,
  &team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_team4_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, srv, PutOnIcecream_Request)() {
  if (!team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_type_support_handle.typesupport_identifier) {
    team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &team4_msgs__srv__PutOnIcecream_Request__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "team4_msgs/srv/detail/put_on_icecream__rosidl_typesupport_introspection_c.h"
// already included above
// #include "team4_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "team4_msgs/srv/detail/put_on_icecream__functions.h"
// already included above
// #include "team4_msgs/srv/detail/put_on_icecream__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  team4_msgs__srv__PutOnIcecream_Response__init(message_memory);
}

void team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_fini_function(void * message_memory)
{
  team4_msgs__srv__PutOnIcecream_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_member_array[1] = {
  {
    "is_okay",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team4_msgs__srv__PutOnIcecream_Response, is_okay),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_members = {
  "team4_msgs__srv",  // message namespace
  "PutOnIcecream_Response",  // message name
  1,  // number of fields
  sizeof(team4_msgs__srv__PutOnIcecream_Response),
  team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_member_array,  // message members
  team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_type_support_handle = {
  0,
  &team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_team4_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, srv, PutOnIcecream_Response)() {
  if (!team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_type_support_handle.typesupport_identifier) {
    team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &team4_msgs__srv__PutOnIcecream_Response__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "team4_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "team4_msgs/srv/detail/put_on_icecream__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_service_members = {
  "team4_msgs__srv",  // service namespace
  "PutOnIcecream",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_Request_message_type_support_handle,
  NULL  // response message
  // team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_Response_message_type_support_handle
};

static rosidl_service_type_support_t team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_service_type_support_handle = {
  0,
  &team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, srv, PutOnIcecream_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, srv, PutOnIcecream_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_team4_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, srv, PutOnIcecream)() {
  if (!team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_service_type_support_handle.typesupport_identifier) {
    team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, srv, PutOnIcecream_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team4_msgs, srv, PutOnIcecream_Response)()->data;
  }

  return &team4_msgs__srv__detail__put_on_icecream__rosidl_typesupport_introspection_c__PutOnIcecream_service_type_support_handle;
}
