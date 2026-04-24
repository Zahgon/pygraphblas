"""Contains all automatically generated Semirings from CFFI.

This documentation does not show all the semirings in this module
because of the sheer number of them (over 1700).  Please see the
SuiteSparse User Guide for more information on the semirings usable in
The GraphBLAS.

All the standard and extension semirings that comes with SuiteSparse
are represented by objects in this module.  For example
`pygraphblas.semiring.PLUS_TIMES_INT64`.

"""

import sys
import re
import contextvars
from itertools import chain
from collections import defaultdict

from .base import lib, ffi, _check
from .monoid import Monoid
from . import types

current_semiring = contextvars.ContextVar("current_semiring")

__all__ = ["Semiring", "current_semiring"]


class Semiring:

    __slots__ = ("name", "semiring", "token", "pls", "mul", "type")

    def __init__(self, pls, mul, typ, semiring, udt=None):
        raise NotImplementedError

    def __call__(self, A, B, *args, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exception_type, exception_value, traceback):
        raise NotImplementedError

    def get_op(self):
        raise NotImplementedError

    @property
    def ztype(self):
        pass

    def print(self, level=2, name="", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`.

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        pass


non_boolean_re = re.compile(
    "^(GxB|GrB)_(MIN|MAX|PLUS|TIMES|ANY)_"
    "(FIRST|FIRSTI|FIRSTJ|FIRSTI1|FIRSTJ1|SECOND|SECONDI|SECONDJ|SECONDI1|SECONDJ1|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|ISEQ|ISNE|"
    "ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR|PAIR)_"
    "(?:SEMIRING_)?"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$"
)

boolean_re = re.compile(
    "^(GxB|GrB)_(LOR|LAND|LXOR|EQ|ANY)_"
    "(EQ|NE|GT|LT|GE|LE)_"
    "(?:SEMIRING_)?"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64|FC32|FC64)$"
)

pure_bool_re = re.compile(
    "^(GxB|GrB)_(LOR|LAND|LXOR|EQ|ANY)_"
    "(FIRST|SECOND|LOR|LAND|LXOR|EQ|GT|LT|GE|LE|PAIR)_"
    "(?:SEMIRING_)?"
    "(BOOL)$"
)

complex_re = re.compile(
    "^(GxB|GrB)_(PLUS|TIMES|ANY)_"
    "(FIRST|SECOND|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|PAIR)_"
    "(?:SEMIRING_)?"
    "(FC32|FC64)$"
)

bitwise_re = re.compile(
    "^(GxB|GrB)_(BOR|BAND|BXOR|BXNOR)_"
    "(BOR|BAND|BXOR|BXNOR)_"
    "(?:SEMIRING_)?"
    "(UINT8|UINT16|UINT32|UINT64)$"
)


def semiring_group(reg):
    raise NotImplementedError


def semiring_template(r):  # pragma: nocover
    pass


def build_semirings(__pdoc__):
    raise NotImplementedError
