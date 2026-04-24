"""Base module containing some utility functions, exceptions and low
level library import.

"""

from pprint import pprint
from suitesparse_graphblas import lib, ffi
from numba import njit

__all__ = [
    "lib",
    "ffi",
    "NULL",
    "GraphBLASException",
    "NoValue",
    "UninitializedObject",
    "InvalidObject",
    "NullPointer",
    "InvalidObject",
    "NullPointer",
    "InvalidValue",
    "InvalidIndex",
    "DomainMismatch",
    "DimensionMismatch",
    "OutputNotEmpty",
    "OutOfMemory",
    "InsufficientSpace",
    "IndexOutOfBound",
    "Panic",
    "options_set",
    "options_get",
    "GxB_IMPLEMENTATION",
    "GxB_SPEC",
    "GxB_INDEX_MAX",
]

NULL = ffi.NULL
GxB_INDEX_MAX = lib.GxB_INDEX_MAX

GxB_IMPLEMENTATION = (
    lib.GxB_IMPLEMENTATION_MAJOR,
    lib.GxB_IMPLEMENTATION_MINOR,
    lib.GxB_IMPLEMENTATION_SUB,
)

GxB_SPEC = (lib.GxB_SPEC_MAJOR, lib.GxB_SPEC_MINOR, lib.GxB_SPEC_SUB)


def options_set(
    nthreads=None,
    chunk=None,
    burble=None,
    hyper_switch=None,
    bitmap_switch=None,
    format=None,
):
    """Set global library options.

    This options are passed directly to SuiteSparse so see the
    SuiteSparse User Guide for details.

    - `nthreads`: Globals number of threads to use.

    - `chunk`: Chunk size for dividing parallel work.

    - `burble`: Switch to enable "burble" debug output.  SuiteSparse
      must be compiled with burble turned on.

    - `hyper_switch`: Controls the hypersparsity of the internal data
      structure for a matrix.  The parameter is typically in the range
      0 to 1.

    - `bitmap_switch`: Controls when to switch to bitmap format.

    - `format`: Default global matrix data format.

    """
    raise NotImplementedError


def options_get():
    """Get global library options.  See SuiteSparse User Guide.

    >>> pprint(options_get())
    {'bitmap_switch': [...],
     'burble': ...,
     'chunk': ...,
     'format': ...,
     'hyper_switch': ...,
     'nthreads': ...}

    """
    raise NotImplementedError


class GraphBLASException(Exception):
    pass


class NoValue(GraphBLASException):
    pass


class UninitializedObject(GraphBLASException):
    pass


class InvalidObject(GraphBLASException):
    pass


class NullPointer(GraphBLASException):
    pass


class InvalidValue(GraphBLASException):
    pass


class InvalidIndex(GraphBLASException):
    pass


class DomainMismatch(GraphBLASException):
    pass


class DimensionMismatch(GraphBLASException):
    pass


class OutputNotEmpty(GraphBLASException):
    pass


class OutOfMemory(GraphBLASException):
    pass


class InsufficientSpace(GraphBLASException):
    pass


class IndexOutOfBound(GraphBLASException):
    pass


class Panic(GraphBLASException):
    pass


_error_codes = {
    1: NoValue,
    2: UninitializedObject,
    3: InvalidObject,
    4: NullPointer,
    5: InvalidValue,
    6: InvalidIndex,
    7: DomainMismatch,
    8: DimensionMismatch,
    9: OutputNotEmpty,
    10: OutOfMemory,
    11: InsufficientSpace,
    12: IndexOutOfBound,
    13: Panic,
}


def _check(res, raise_no_val=False):
    raise NotImplementedError


_all_slice = slice(None, None, None)


def _build_range(rslice, stop_val):
    # if already a list, return it and its length
    raise NotImplementedError


def _get_select_op(op):
    raise NotImplementedError


def _get_bin_op(op, funcs):
    raise NotImplementedError
