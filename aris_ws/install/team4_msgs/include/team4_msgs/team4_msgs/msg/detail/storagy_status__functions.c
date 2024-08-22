// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from team4_msgs:msg/StoragyStatus.idl
// generated code does not contain a copyright notice
#include "team4_msgs/msg/detail/storagy_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `storagy_status`
#include "rosidl_runtime_c/string_functions.h"

bool
team4_msgs__msg__StoragyStatus__init(team4_msgs__msg__StoragyStatus * msg)
{
  if (!msg) {
    return false;
  }
  // storagy_status
  if (!rosidl_runtime_c__String__init(&msg->storagy_status)) {
    team4_msgs__msg__StoragyStatus__fini(msg);
    return false;
  }
  // seat_number
  return true;
}

void
team4_msgs__msg__StoragyStatus__fini(team4_msgs__msg__StoragyStatus * msg)
{
  if (!msg) {
    return;
  }
  // storagy_status
  rosidl_runtime_c__String__fini(&msg->storagy_status);
  // seat_number
}

bool
team4_msgs__msg__StoragyStatus__are_equal(const team4_msgs__msg__StoragyStatus * lhs, const team4_msgs__msg__StoragyStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // storagy_status
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->storagy_status), &(rhs->storagy_status)))
  {
    return false;
  }
  // seat_number
  if (lhs->seat_number != rhs->seat_number) {
    return false;
  }
  return true;
}

bool
team4_msgs__msg__StoragyStatus__copy(
  const team4_msgs__msg__StoragyStatus * input,
  team4_msgs__msg__StoragyStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // storagy_status
  if (!rosidl_runtime_c__String__copy(
      &(input->storagy_status), &(output->storagy_status)))
  {
    return false;
  }
  // seat_number
  output->seat_number = input->seat_number;
  return true;
}

team4_msgs__msg__StoragyStatus *
team4_msgs__msg__StoragyStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__msg__StoragyStatus * msg = (team4_msgs__msg__StoragyStatus *)allocator.allocate(sizeof(team4_msgs__msg__StoragyStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(team4_msgs__msg__StoragyStatus));
  bool success = team4_msgs__msg__StoragyStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
team4_msgs__msg__StoragyStatus__destroy(team4_msgs__msg__StoragyStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    team4_msgs__msg__StoragyStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
team4_msgs__msg__StoragyStatus__Sequence__init(team4_msgs__msg__StoragyStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__msg__StoragyStatus * data = NULL;

  if (size) {
    data = (team4_msgs__msg__StoragyStatus *)allocator.zero_allocate(size, sizeof(team4_msgs__msg__StoragyStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = team4_msgs__msg__StoragyStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        team4_msgs__msg__StoragyStatus__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
team4_msgs__msg__StoragyStatus__Sequence__fini(team4_msgs__msg__StoragyStatus__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      team4_msgs__msg__StoragyStatus__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

team4_msgs__msg__StoragyStatus__Sequence *
team4_msgs__msg__StoragyStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__msg__StoragyStatus__Sequence * array = (team4_msgs__msg__StoragyStatus__Sequence *)allocator.allocate(sizeof(team4_msgs__msg__StoragyStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = team4_msgs__msg__StoragyStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
team4_msgs__msg__StoragyStatus__Sequence__destroy(team4_msgs__msg__StoragyStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    team4_msgs__msg__StoragyStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
team4_msgs__msg__StoragyStatus__Sequence__are_equal(const team4_msgs__msg__StoragyStatus__Sequence * lhs, const team4_msgs__msg__StoragyStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!team4_msgs__msg__StoragyStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
team4_msgs__msg__StoragyStatus__Sequence__copy(
  const team4_msgs__msg__StoragyStatus__Sequence * input,
  team4_msgs__msg__StoragyStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(team4_msgs__msg__StoragyStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    team4_msgs__msg__StoragyStatus * data =
      (team4_msgs__msg__StoragyStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!team4_msgs__msg__StoragyStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          team4_msgs__msg__StoragyStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!team4_msgs__msg__StoragyStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
