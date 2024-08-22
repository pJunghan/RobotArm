// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from team4_msgs:srv/PutOnIcecream.idl
// generated code does not contain a copyright notice
#include "team4_msgs/srv/detail/put_on_icecream__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
team4_msgs__srv__PutOnIcecream_Request__init(team4_msgs__srv__PutOnIcecream_Request * msg)
{
  if (!msg) {
    return false;
  }
  // seat_number
  return true;
}

void
team4_msgs__srv__PutOnIcecream_Request__fini(team4_msgs__srv__PutOnIcecream_Request * msg)
{
  if (!msg) {
    return;
  }
  // seat_number
}

bool
team4_msgs__srv__PutOnIcecream_Request__are_equal(const team4_msgs__srv__PutOnIcecream_Request * lhs, const team4_msgs__srv__PutOnIcecream_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // seat_number
  if (lhs->seat_number != rhs->seat_number) {
    return false;
  }
  return true;
}

bool
team4_msgs__srv__PutOnIcecream_Request__copy(
  const team4_msgs__srv__PutOnIcecream_Request * input,
  team4_msgs__srv__PutOnIcecream_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // seat_number
  output->seat_number = input->seat_number;
  return true;
}

team4_msgs__srv__PutOnIcecream_Request *
team4_msgs__srv__PutOnIcecream_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__srv__PutOnIcecream_Request * msg = (team4_msgs__srv__PutOnIcecream_Request *)allocator.allocate(sizeof(team4_msgs__srv__PutOnIcecream_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(team4_msgs__srv__PutOnIcecream_Request));
  bool success = team4_msgs__srv__PutOnIcecream_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
team4_msgs__srv__PutOnIcecream_Request__destroy(team4_msgs__srv__PutOnIcecream_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    team4_msgs__srv__PutOnIcecream_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
team4_msgs__srv__PutOnIcecream_Request__Sequence__init(team4_msgs__srv__PutOnIcecream_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__srv__PutOnIcecream_Request * data = NULL;

  if (size) {
    data = (team4_msgs__srv__PutOnIcecream_Request *)allocator.zero_allocate(size, sizeof(team4_msgs__srv__PutOnIcecream_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = team4_msgs__srv__PutOnIcecream_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        team4_msgs__srv__PutOnIcecream_Request__fini(&data[i - 1]);
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
team4_msgs__srv__PutOnIcecream_Request__Sequence__fini(team4_msgs__srv__PutOnIcecream_Request__Sequence * array)
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
      team4_msgs__srv__PutOnIcecream_Request__fini(&array->data[i]);
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

team4_msgs__srv__PutOnIcecream_Request__Sequence *
team4_msgs__srv__PutOnIcecream_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__srv__PutOnIcecream_Request__Sequence * array = (team4_msgs__srv__PutOnIcecream_Request__Sequence *)allocator.allocate(sizeof(team4_msgs__srv__PutOnIcecream_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = team4_msgs__srv__PutOnIcecream_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
team4_msgs__srv__PutOnIcecream_Request__Sequence__destroy(team4_msgs__srv__PutOnIcecream_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    team4_msgs__srv__PutOnIcecream_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
team4_msgs__srv__PutOnIcecream_Request__Sequence__are_equal(const team4_msgs__srv__PutOnIcecream_Request__Sequence * lhs, const team4_msgs__srv__PutOnIcecream_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!team4_msgs__srv__PutOnIcecream_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
team4_msgs__srv__PutOnIcecream_Request__Sequence__copy(
  const team4_msgs__srv__PutOnIcecream_Request__Sequence * input,
  team4_msgs__srv__PutOnIcecream_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(team4_msgs__srv__PutOnIcecream_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    team4_msgs__srv__PutOnIcecream_Request * data =
      (team4_msgs__srv__PutOnIcecream_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!team4_msgs__srv__PutOnIcecream_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          team4_msgs__srv__PutOnIcecream_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!team4_msgs__srv__PutOnIcecream_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
team4_msgs__srv__PutOnIcecream_Response__init(team4_msgs__srv__PutOnIcecream_Response * msg)
{
  if (!msg) {
    return false;
  }
  // is_okay
  return true;
}

void
team4_msgs__srv__PutOnIcecream_Response__fini(team4_msgs__srv__PutOnIcecream_Response * msg)
{
  if (!msg) {
    return;
  }
  // is_okay
}

bool
team4_msgs__srv__PutOnIcecream_Response__are_equal(const team4_msgs__srv__PutOnIcecream_Response * lhs, const team4_msgs__srv__PutOnIcecream_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // is_okay
  if (lhs->is_okay != rhs->is_okay) {
    return false;
  }
  return true;
}

bool
team4_msgs__srv__PutOnIcecream_Response__copy(
  const team4_msgs__srv__PutOnIcecream_Response * input,
  team4_msgs__srv__PutOnIcecream_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // is_okay
  output->is_okay = input->is_okay;
  return true;
}

team4_msgs__srv__PutOnIcecream_Response *
team4_msgs__srv__PutOnIcecream_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__srv__PutOnIcecream_Response * msg = (team4_msgs__srv__PutOnIcecream_Response *)allocator.allocate(sizeof(team4_msgs__srv__PutOnIcecream_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(team4_msgs__srv__PutOnIcecream_Response));
  bool success = team4_msgs__srv__PutOnIcecream_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
team4_msgs__srv__PutOnIcecream_Response__destroy(team4_msgs__srv__PutOnIcecream_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    team4_msgs__srv__PutOnIcecream_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
team4_msgs__srv__PutOnIcecream_Response__Sequence__init(team4_msgs__srv__PutOnIcecream_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__srv__PutOnIcecream_Response * data = NULL;

  if (size) {
    data = (team4_msgs__srv__PutOnIcecream_Response *)allocator.zero_allocate(size, sizeof(team4_msgs__srv__PutOnIcecream_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = team4_msgs__srv__PutOnIcecream_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        team4_msgs__srv__PutOnIcecream_Response__fini(&data[i - 1]);
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
team4_msgs__srv__PutOnIcecream_Response__Sequence__fini(team4_msgs__srv__PutOnIcecream_Response__Sequence * array)
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
      team4_msgs__srv__PutOnIcecream_Response__fini(&array->data[i]);
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

team4_msgs__srv__PutOnIcecream_Response__Sequence *
team4_msgs__srv__PutOnIcecream_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team4_msgs__srv__PutOnIcecream_Response__Sequence * array = (team4_msgs__srv__PutOnIcecream_Response__Sequence *)allocator.allocate(sizeof(team4_msgs__srv__PutOnIcecream_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = team4_msgs__srv__PutOnIcecream_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
team4_msgs__srv__PutOnIcecream_Response__Sequence__destroy(team4_msgs__srv__PutOnIcecream_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    team4_msgs__srv__PutOnIcecream_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
team4_msgs__srv__PutOnIcecream_Response__Sequence__are_equal(const team4_msgs__srv__PutOnIcecream_Response__Sequence * lhs, const team4_msgs__srv__PutOnIcecream_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!team4_msgs__srv__PutOnIcecream_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
team4_msgs__srv__PutOnIcecream_Response__Sequence__copy(
  const team4_msgs__srv__PutOnIcecream_Response__Sequence * input,
  team4_msgs__srv__PutOnIcecream_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(team4_msgs__srv__PutOnIcecream_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    team4_msgs__srv__PutOnIcecream_Response * data =
      (team4_msgs__srv__PutOnIcecream_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!team4_msgs__srv__PutOnIcecream_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          team4_msgs__srv__PutOnIcecream_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!team4_msgs__srv__PutOnIcecream_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
