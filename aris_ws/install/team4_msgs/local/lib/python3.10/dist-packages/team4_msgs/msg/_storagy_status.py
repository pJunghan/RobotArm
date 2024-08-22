# generated from rosidl_generator_py/resource/_idl.py.em
# with input from team4_msgs:msg/StoragyStatus.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_StoragyStatus(type):
    """Metaclass of message 'StoragyStatus'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('team4_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'team4_msgs.msg.StoragyStatus')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__storagy_status
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__storagy_status
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__storagy_status
            cls._TYPE_SUPPORT = module.type_support_msg__msg__storagy_status
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__storagy_status

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class StoragyStatus(metaclass=Metaclass_StoragyStatus):
    """Message class 'StoragyStatus'."""

    __slots__ = [
        '_storagy_status',
        '_seat_number',
    ]

    _fields_and_field_types = {
        'storagy_status': 'string',
        'seat_number': 'int32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.storagy_status = kwargs.get('storagy_status', str())
        self.seat_number = kwargs.get('seat_number', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.storagy_status != other.storagy_status:
            return False
        if self.seat_number != other.seat_number:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def storagy_status(self):
        """Message field 'storagy_status'."""
        return self._storagy_status

    @storagy_status.setter
    def storagy_status(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'storagy_status' field must be of type 'str'"
        self._storagy_status = value

    @builtins.property
    def seat_number(self):
        """Message field 'seat_number'."""
        return self._seat_number

    @seat_number.setter
    def seat_number(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'seat_number' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'seat_number' field must be an integer in [-2147483648, 2147483647]"
        self._seat_number = value
