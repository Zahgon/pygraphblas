"""GraphBLAS Scalar (SuiteSparse Only)

"""
from .base import (
    lib,
    ffi,
    NULL,
    _check,
)

from .types import _gb_from_type

__all__ = ["Scalar"]


class Scalar:
    """GraphBLAS Scalar

    Used for now mostly for the `pygraphblas.Matrix.select`.

    """

    __slots__ = ("_scalar", "type")

    def __init__(self, s, typ):
        raise NotImplementedError

    def __del__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def dup(self):
        """Create an duplicate Scalar from the given argument."""
        pass

    @classmethod
    def from_type(cls, typ):
        """Create an empty Scalar from the given type and size."""
        pass

    @classmethod
    def from_value(cls, value):
        """Create an empty Scalar from the given type and size."""
        pass

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Scalar."""
        pass

    def clear(self):
        """Clear the scalar."""
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __setitem__(self, index, value):
        raise NotImplementedError

    def wait(self):
        pass

    @property
    def nvals(self):
        """Return the number of values in the scalar (0 or 1)."""
        pass

    def __bool__(self):
        raise NotImplementedError
