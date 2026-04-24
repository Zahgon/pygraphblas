"""Contains all automatically generated Monoids from CFFI.

The scalar addition of conventional matrix multiplication is replaced
with a *monoid*.  A monoid is an associative and commutative binary
operator `z=f(x,y)` where all three domains are the same (the
types of `x`, `y`, and `z`), and where the operator has
an identity value `id` such that `f(x,id)=f(id,x)=x`.
Performing matrix multiplication with a semiring uses a monoid in
place of the `add` operator, scalar addition being just one of many
possible monoids.  The identity value of addition is zero, since
$x+0=0+x=x$.  GraphBLAS includes many built-in operators suitable for
use as a monoid: min (with an identity value of positive infinity),
max (whose identity is negative infinity), add (identity is zero),
multiply (with an identity of one), four logical operators: AND, OR,
exclusive-OR, and Boolean equality (XNOR), four bitwise operators
(AND, OR, XOR, and XNOR), and the ANY operator.  User-created monoids
can be defined with any associative and commutative operator that has
an identity value.
"""

__all__ = ["Monoid", "current_monoid"]

import os
import sys
import re
import contextvars
from itertools import chain
from collections import defaultdict

from .base import lib, ffi, _check
from .binaryop import BinaryOp
from . import types

current_monoid = contextvars.ContextVar("current_monoid")


class Monoid:

    __slots__ = ("name", "monoid", "token", "op", "type")

    def __init__(self, op, typ, monoid, udt=None, boolean=False):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *errors):
        raise NotImplementedError

    def __call__(self, A, B, *args, **kwargs):
        raise NotImplementedError

    def get_op(self):
        raise NotImplementedError

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


gxb_monoid_re = re.compile(
    "^GxB_(MIN|MAX|PLUS|TIMES|ANY|BOR|BAND|BXOR|BXNOR)_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64|FC32|FC64)_MONOID$"
)

grb_monoid_re = re.compile(
    "^GrB_(MIN|MAX|PLUS|TIMES)_MONOID_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64|FC32|FC64)$"
)

pure_bool_re = re.compile("^GxB_(ANY|LOR|LAND|LXOR|LXNOR|EQ)_(BOOL)_MONOID$")
pure_bool_re_v13 = re.compile("^GrB_(LOR|LAND|LXOR|LXNOR)_MONOID_(BOOL)$")


def monoid_group(reg):
    raise NotImplementedError


def build_monoids(__pdoc__):
    raise NotImplementedError
