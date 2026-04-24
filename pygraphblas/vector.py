"""High level wrapper around GraphBLAS Vectors.

"""
import random
import sys
import operator
import weakref
from array import array
from functools import partial
import numpy as np

from .base import (
    lib,
    ffi,
    NULL,
    NoValue,
    _check,
    _error_codes,
    _get_bin_op,
    _get_select_op,
    _build_range,
    GxB_INDEX_MAX,
)
from . import types
from .scalar import Scalar
from .semiring import current_semiring, Semiring
from .binaryop import current_accum, current_binop, Accum
from .monoid import current_monoid, Monoid
from . import descriptor
from .descriptor import Descriptor, T1, current_desc


__all__ = ["Vector"]
__pdoc__ = {"Vector.__init__": False}


class Vector:
    """GraphBLAS Sparse Vector

    This is a high-level wrapper around the low-level GrB_Vector type.

    A Vector supports many possible operations according to the
    GraphBLAS API.  Many of those operations have overloaded
    operators.

    Operator | Description | Default
    --- | --- | ---
    v @    A | Vector Vector Multiplication | type default PLUS_TIMES semiring
    v @=   A | In-place Vector Vector Multiplication | type default PLUS_TIMES semiring
    v \\|  w | Vector Union | type default SECOND combiner
    v \\|= w | In-place Vector Union | type default SECOND combiner
    v &    w | Vector Intersection | type default SECOND combiner
    v &=   w | In-place Vector Intersection | type default SECOND combiner
    v +    w | Vector Element-Wise Union | type default PLUS combiner
    v +=   w | In-place Vector Element-Wise Union | type default PLUS combiner
    v -    w | Vector Element-Wise Union | type default MINUS combiner
    v -=   w | In-place Vector Element-Wise Union | type default MINUS combiner
    v *    w | Vector Element-Wise Intersection | type default TIMES combiner
    v *=   w | In-place Vector Element-Wise Intersection | type default TIMES combiner
    v /    w | Vector Element-Wise Intersection | type default DIV combiner
    v /=   w | In-place Vector Element-Wise Intersection | type default DIV combiner
    v ==   w | Compare Element-Wise Union | type default EQ operator
    v !=   w | Compare Element-Wise Union | type default NE operator
    v <    w | Compare Element-Wise Union | type default LT operator
    v >    w | Compare Element-Wise Union | type default GT operator
    v <=   w | Compare Element-Wise Union | type default LE operator
    v >=   w | Compare Element-Wise Union | type default GE operator

    Note that all the above operator syntax is mearly sugar over
    various combinations of calling `Matrix.mxv`, `Vector.vxm`,
    `Vector.eadd`, and `Vector.emult`.

    """

    __slots__ = ("_vector", "type", "_keep_alives")

    def _check(self, res):
        raise NotImplementedError

    def __init__(self, vec, typ=None):
        raise NotImplementedError

    def __del__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getattr__(self, name):
        """Look up operators as attributes for the given object."""
        raise NotImplementedError

    @property
    def indices(self):
        """cdata array of vector indexes.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.indices)
        [0, 1, 2]

        """
        pass

    @property
    def I(self):
        """Iterator over for `Vector.indices`.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.I)
        [0, 1, 2]

        """
        pass

    @property
    def npI(self):
        """numpy array over `Vector.indices`.

        >>> v = Vector.from_1_to_n(3)
        >>> v.npI
        array([0, 1, 2], dtype=uint64)

        """
        pass

    @property
    def vals(self):
        """Iterator of vector values.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.vals)
        [1, 2, 3]

        """
        pass

    @property
    def V(self):
        """Iterator over for `Vector.vals`.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.V)
        [1, 2, 3]

        """
        pass

    @property
    def npV(self):
        """numpy array over `Vector.vals`.

        >>> v = Vector.from_1_to_n(3)
        >>> v.npV
        array([1, 2, 3])

        """
        pass

    def all(self, other, op):
        """Do all elements in self compare True with op to other?

        >>> from . import INT64
        >>> M = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> N = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> O = Vector.from_lists([0, 1], [1, 2])
        >>> P = Vector.from_lists([0, 1], [1, 2], size=3)
        >>> Q = Vector.from_lists([0, 1, 3], [1, 2, 3])
        >>> assert M.all(N, INT64.eq)
        >>> assert not M.all(N, INT64.gt)
        >>> assert not M.all(O, INT64.eq)
        >>> assert not M.all(P, INT64.eq)
        >>> assert not M.all(Q, INT64.eq)

        """
        pass

    def iseq(self, other, eq_op=None):
        """Compare two vectors for equality.

        Note to be confused with the `==` operator which does
        element-wise comparison and returns a `Vector`.

        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> w = Vector.from_lists([0,1], [1, 1])
        >>> x = Vector.from_lists([0,1], [1.0, 1.0])
        >>> v.iseq(w)
        True

        >>> v.iseq(w, eq_op=types.UINT64.GE)
        True

        >>> v.iseq(x)
        False
        """
        pass

    def isne(self, other):
        """Compare two vectors for inequality.
        Note to be confused with the `==` operator which does
        element-wise comparison and returns a `Vector`.

        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> w = Vector.from_lists([0,1], [1, 1])
        >>> v.isne(w)
        False

        """
        pass

    @classmethod
    def sparse(cls, typ, size=None, fill=None, mask=None):
        """Create an empty Vector from the given type.  If `size` is not
        specified it defaults to `pygraphblas.GxB_INDEX_MAX`.

        >>> v = Vector.sparse(types.INT64, 3)
        >>> v
        <Vector(INT64 size: 3, nvals: 0)>
        >>> v.size
        3
        >>> v = Vector.sparse(types.INT64)
        >>> v
        <Vector(INT64, nvals: 0)>
        >>> v.size == lib.GxB_INDEX_MAX
        True

        >>> v[42] = True
        >>> w = Vector.sparse(types.INT64, fill=42, mask=v)
        >>> list(w)
        [(42, 42)]

        If no `fill` is provided, the `type.default_zero` is used:

        >>> w = Vector.sparse(types.INT64, mask=v)
        >>> list(w)
        [(42, 0)]
        """
        raise NotImplementedError

    @classmethod
    def random(
        cls,
        typ,
        nvals,
        size=lib.GxB_INDEX_MAX,
        make_pattern=False,
        seed=None,
    ):  # pragma: nocover
        """ """
        pass

    @classmethod
    def from_lists(cls, I, V, size=None, typ=None):
        """Create a new vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max values of
        the provides size indices.

        If the second argument is a scalar value, an "iso" vector is
        created where all values equal that scalar.

        >>> v = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> w = Vector.from_lists([0, 1, 2], True)
        >>> assert not v.iseq(w)
        >>> assert v.pattern().iseq(w)
        """
        pass

    @classmethod
    def from_list(cls, I):
        """Create a new dense vector from the given lists of values."""
        pass

    @classmethod
    def from_1_to_n(cls, n):
        """Generate a vector from 1 to n.

        >>> v = Vector.from_1_to_n(3)
        >>> print(v)
        0| 1
        1| 2
        2| 3
        """
        pass

    def dup(self):
        """Create an duplicate Vector from the given argument.

        >>> v = Vector.from_1_to_n(3)
        >>> w = v.dup()
        >>> w is not v
        True
        >>> w.iseq(v)
        True
        >>> print(w)
        0| 1
        1| 2
        2| 3
        """
        pass

    @property
    def hyper_switch(self):  # pragma: nocover
        """Get the hyper_switch threshold. (See SuiteSparse User Guide)"""
        pass

    @hyper_switch.setter
    def hyper_switch(self, switch):  # pragma: nocover
        """Set the hyper_switch threshold. (See SuiteSparse User Guide)"""
        pass

    @property
    def sparsity(self):  # pragma: nocover
        """Get Vector sparsity control. (See SuiteSparse User Guide)"""
        pass

    @sparsity.setter
    def sparsity(self, sparsity):  # pragma: nocover
        """Set Vector sparsity control. (See SuiteSparse User Guide)"""
        pass

    @property
    def sparsity_status(self):  # pragma: nocover
        """Get Vector sparsity status. (See SuiteSparse User Guide)"""
        pass

    @classmethod
    def dense(cls, typ, size=None, fill=None):
        """Return a dense vector of `typ` and `size`.  If `fill` is provided,
        use that value otherwise use `self.type.default_zero`

        >>> print(Vector.dense(types.FP32, 3))
        0|0.0
        1|0.0
        2|0.0
        >>> print(Vector.dense(types.FP32, 3, fill=42.0))
        0|42.0
        1|42.0
        2|42.0

        """
        pass

    @classmethod
    def iso(cls, value, size=GxB_INDEX_MAX):
        """Build an "iso" vector from a scalar value.

        This is similar to `Vector.dense` but infers the type of the
        new Matrix from the provided vbalue.

        >>> v = Vector.iso(3)
        >>> assert v[42] == 3
        """
        pass

    def to_lists(self):
        """Extract the indices and values of the Vector as 2 lists.

        >>> Vector.from_1_to_n(3).to_lists()
        [[0, 1, 2], [1, 2, 3]]

        """
        pass

    def to_arrays(self):
        """Return as python `array` objects.

        >>> Vector.from_1_to_n(3).to_arrays()
        (array('L', [0, 1, 2]), array('q', [1, 2, 3]))

        """
        pass

    @property
    def size(self):
        """Return the size of the vector.

        >>> Vector.from_1_to_n(3).size
        3

        """
        pass

    @property
    def nvals(self):
        """Return the number of values in the vector.

        >>> v = Vector.from_1_to_n(3)
        >>> v.nvals
        3
        >>> v.clear()
        >>> v.nvals
        0

        """
        pass

    @property
    def memory_usage(self):
        """Returns the memory usage of the Vector.

        >>> v = Vector.from_lists([0, 1, 2], [1, 2, 0])
        >>> assert v.memory_usage > 0
        """
        pass

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Vector."""
        pass

    def _full(self):
        pass

    def _compare(self, other, op, strop):
        pass

    def __gt__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def eadd(
        self,
        other,
        add_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Element-wise addition with other vector.

        Element-wise addition applies a binary operator element-wise
        on two vectors `v` and `w`, for all entries that appear in the
        set union of the patterns of `A` and `B`.

        The only difference between element-wise multiplication and
        addition is the pattern of the result, and what happens to
        entries outside the intersection. With multiplication the
        pattern of T is the intersection; with addition it is the set
        union. Entries outside the set intersection are dropped for
        multiplication, and kept for addition; in both cases the
        operator is only applied to those (and only those) entries in
        the intersection. Any binary operator can be used
        interchangeably for either operation.

        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> V = list(range(len(I)))
        >>> v = Vector.from_lists(I, V, 7)

        >>> w = Vector.from_lists(
        ...    [0, 1, 4, 6],
        ...    [9, 1, 4, 7], 7)

        >>> print(v.eadd(w))
        0|10
        1| 4
        2| 4
        3| 6
        4|11
        5| 8
        6|18

        This can also be accomplished with the `+` operators:

        >>> print(v + w)
        0|10
        1| 4
        2| 4
        3| 6
        4|11
        5| 8
        6|18

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MIN:
        ...     print(v + w)
        0| 1
        1| 1
        2| 4
        3| 6
        4| 4
        5| 8
        6| 7

        You can provide a monoid for the operation:

        >>> print(v.eadd(w, v.type.min_monoid))
        0| 1
        1| 1
        2| 4
        3| 6
        4| 4
        5| 8
        6| 7

        Or you can use a semiring:

        >>> print(v.eadd(w, v.type.min_plus))
        0| 1
        1| 1
        2| 4
        3| 6
        4| 4
        5| 8
        6| 7

        The following operators default to use `eadd`:

        Operator | Description | Default
        --- | --- | ---
        v \\|  w | Vector Union | type default SECOND combiner
        v \\|= w | In-place Vector Union | type default SECOND combiner
        v +    w | Vector Element-Wise Union | type default PLUS combiner
        v +=   w | In-place Vector Element-Wise Union | type default PLUS combiner
        v -    w | Vector Element-Wise Union | type default MINUS combiner
        v -=   w | In-place Vector Element-Wise Union | type default MINUS combiner

        """
        pass

    def emult(
        self,
        other,
        mult_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Element-wise multiplication with other vector.

        Element-wise multiplication applies a binary operator
        element-wise on two vectors A and B, for all entries that
        appear in the set intersection of the patterns of A and B.

        The only difference between element-wise multiplication and
        addition is the pattern of the result, and what happens to
        entries outside the intersection. With multiplication the
        pattern of T is the intersection; with addition it is the set
        union. Entries outside the set intersection are dropped for
        multiplication, and kept for addition; in both cases the
        operator is only applied to those (and only those) entries in
        the intersection. Any binary operator can be used
        interchangeably for either operation.

        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> V = list(range(len(I)))
        >>> v = Vector.from_lists(I, V, 7)

        >>> w = Vector.from_lists(
        ...    [0, 1, 4, 6],
        ...    [9, 1, 4, 7], 7)

        >>> print(v.emult(w))
        0| 9
        1| 3
        2|
        3|
        4|28
        5|
        6|77

        This can also be accomplished with the `+` operators:

        >>> print(v * w)
        0| 9
        1| 3
        2|
        3|
        4|28
        5|
        6|77

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MAX:
        ...     print(v * w)
        0| 9
        1| 3
        2|
        3|
        4|28
        5|
        6|77

        """
        pass

    def vxm(
        self,
        other,
        semiring=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Vector-Matrix multiply.


        Multiply this row vector by `other` matrix "on the left".  For
        column matrix/vector multiplication "on the right" see
        `Matrix.mxv`.

        `vxm` can also be called directly or with the `@` operator:

        >>> from . import Matrix, INT64
        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> v = Vector.from_lists([0, 1, 2], [2, 3, 4])
        >>> o = v.vxm(M)
        >>> print(o)
        0|12
        1| 2
        2| 6
        >>> o = v @ M
        >>> print(o)
        0|12
        1| 2
        2| 6

        By default, `mxv` and `@` create a new result matrix of the
        correct type and dimensions if one is not provided.  If you
        want to provide your own matrix to put the result in, you can
        pass it in the `out` parameter.  This is useful for
        accumulating results into a single matrix with minimal
        copying.  This is also supported by the `@=` syntax:

        >>> o = v.dup()
        >>> v.vxm(M, accum=INT64.plus, out=o) is o
        True
        >>> print(o)
        0|14
        1| 5
        2|10
        >>> o = v.dup()
        >>> with Accum(INT64.min):
        ...     o @= M
        >>> print(o)
        0| 2
        1| 2
        2| 4

        The default semiring depends on the infered result type.  In
        the case of numbers, the default semiring is `PLUS_TIMES`.  In
        the case of type `BOOL`, it is `BOOL.lor_land`.

        >>> o = v.vxm(M, semiring=INT64.min_plus)
        >>> print(o)
        0| 7
        1| 3
        2| 5

        An explicit semiring can be passed to the method or provided
        with a context manager:

        >>> with INT64.min_plus:
        ...     o = v @ M
        >>> print(o)
        0| 7
        1| 3
        2| 5

        Or the semiring can be accessed via an attribute on the
        vector:

        >>> o = v.min_plus(M)
        >>> print(o)
        0| 7
        1| 3
        2| 5

        Descriptors and accumulators can also be provided as an
        argument or a context manager:

        >>> o = v.vxm(M, desc=descriptor.T0)
        >>> print(o)
        0|12
        1| 2
        2| 6
        >>> with descriptor.T0:
        ...     o = v @ M
        >>> print(o)
        0|12
        1| 2
        2| 6
        >>> del o[1]
        >>> o = v.vxm(M, mask=o)
        >>> print(o)
        0|12
        1|
        2| 6

        """
        pass

    def __matmul__(self, other):
        raise NotImplementedError

    def __imatmul__(self, other):
        raise NotImplementedError

    def __and__(self, other):
        raise NotImplementedError

    def __iand__(self, other):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def __ior__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __iadd__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError

    def __invert__(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __abs__(self):
        raise NotImplementedError

    def clear(self):
        """Clear this vector removing all entries."""
        pass

    def resize(self, size=lib.GxB_INDEX_MAX):
        """Resize the vector.  If the dimensions decrease, entries that fall
        outside the resized vector are deleted.

        >>> v = Vector.dense(types.UINT8, 2)
        >>> v.resize(3)
        >>> print(v)
        0| 0
        1| 0
        2|

        """
        pass

    def _get_args(self, mask=None, accum=None, desc=None):
        raise NotImplementedError

    def reduce(self, mon=None, accum=None, desc=None):
        """Do a scalar reduce based on this object's type:

        >>> V = Vector.random(types.UINT8, 10, 3, seed=42)
        >>> V.reduce()
        114

        >>> V = Vector.random(types.FP32, 10, 3, seed=42)
        >>> V.reduce()
        0.9517456293106079

        >>> V = Vector.random(types.UINT8, 10, 3, seed=42)
        >>> V.reduce(V.type.min_monoid)
        13

        >>> V = Vector.random(types.BOOL, 10, 3, seed=42)
        >>> V.reduce()
        False

        """
        pass

    def reduce_bool(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce vector to a boolean.

        >>> v = Vector.from_lists([0, 1], [True, False])
        >>> v.reduce_bool()
        True
        >>> v[0] = False
        >>> v.reduce_bool()
        False
        >>> v[1] = True
        >>> v.reduce_bool(types.BOOL.LAND_MONOID)
        False

        """
        pass

    def reduce_int(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce vector to a integer.

        >>> v = Vector.from_lists([0, 1], [1, 1])
        >>> v.reduce_int()
        2
        >>> v[0] = 0
        >>> v.reduce_int()
        1
        >>> v[1] = 2
        >>> v.reduce_int(types.INT64.MIN_MONOID)
        0

        """
        pass

    def reduce_float(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce vector to a float.

        >>> v = Vector.from_lists([0, 1], [1.2, 1.1])
        >>> v.reduce_float()
        2.3
        >>> v[0] = 0
        >>> v.reduce_float()
        1.1
        >>> v[1] = 2.2
        >>> v.reduce_float(types.FP64.MIN_MONOID)
        0.0

        """
        pass

    def max(self):
        """Return the max of the vector.

        >>> M = Vector.from_lists([0, 1, 2], [False, False, False])
        >>> M.max()
        False
        >>> M = Vector.from_lists([0, 1, 2], [False, False, True])
        >>> M.max()
        True
        >>> M = Vector.from_lists([0, 1, 2], [-42, 0, 149])
        >>> M.max()
        149
        >>> M = Vector.from_lists([0, 1, 2], [-42.0, 0.0, 149.0])
        >>> M.max()
        149.0
        >>> M = Vector.from_lists([0], [1j])
        >>> M.max()
        Traceback (most recent call last):
        ...
        TypeError: Un-maxable type
        """
        pass

    def min(self):
        """Return the min of the vector.

        >>> M = Vector.from_lists([0, 1, 2], [True, True, True])
        >>> M.min()
        True
        >>> M = Vector.from_lists([0, 1, 2], [False, True, True])
        >>> M.min()
        False
        >>> M = Vector.from_lists([0, 1, 2], [-42, 0, 149])
        >>> M.min()
        -42
        >>> M = Vector.from_lists([0, 1, 2], [-42.0, 0.0, 149.0])
        >>> M.min()
        -42.0
        >>> M = Vector.from_lists([0], [1j])
        >>> M.min()
        Traceback (most recent call last):
        ...
        TypeError: Un-minable type
        """
        pass

    def apply(self, op, out=None, mask=None, accum=None, desc=None):
        """Apply Unary op to vector elements.
        >>> from . import UINT64
        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> print(v.apply(UINT64.ainv))
        0|-1
        1|-1

        Unary operators can also be accessed by atribute name on
        vectors they are applied to:

        >>> print(v.ainv())
        0|-1
        1|-1

        """
        raise NotImplementedError

    def apply_first(self, first, op, out=None, mask=None, accum=None, desc=None):
        """Apply a binary operator to the entries in a vector, binding the first input
        to a scalar first.


        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> print(v.apply_first(3, types.UINT64.PLUS))
        0| 4
        1| 4
        >>> w = Vector.sparse(v.type, v.size)
        >>> v.apply_first(3, types.UINT64.PLUS, out=w) is w
        True

        """
        pass

    def apply_second(self, op, second, out=None, mask=None, accum=None, desc=None):
        """Apply a binary operator to the entries in a vector, binding the second input
        to a scalar second.

        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> print(v.apply_second(types.UINT64.PLUS, 3))
        0| 4
        1| 4
        >>> w = Vector.sparse(v.type, v.size)
        >>> v.apply_second(types.UINT64.PLUS, 3, out=w) is w
        True
        >>> u = Vector.from_lists([0,1], [1.1, 2.2])
        >>> u.apply_second(u.type.TIMES, 3.3, out=u) is u
        True
        >>> u = Vector.from_lists([0,1], [1.1, 2.2])
        >>> print(u * 3)
        0|3.3
        1|6.6
        >>> x = Vector.from_lists([0,1], [1.1, 2.2])
        >>> x *= 3.0
        >>> print(x)
        0|3.3
        1|6.6

        """
        pass

    def select(self, op, thunk=None, out=None, mask=None, accum=None, desc=None):
        """Select elements that match the given select operation condition.
        See `Matrix.select` for possible operators.

        >>> v = Vector.from_lists([0,1], [1, 0])
        >>> print(v.select('>', 0))
        0| 1
        1|

        >>> w = Vector.sparse(types.UINT8, 2)
        >>> v.select('>', 0, out=w) is w
        True

        `min` and `max` selectors can be shortcuts for selecting all
        elements that equal the min or max reduction of all elements.

        >>> print(v.select('min'))
        0|
        1| 0
        >>> print(v.select('max'))
        0| 1
        1|

        """
        pass

    def pattern(self, typ=types.BOOL):
        """Return the pattern of the vector, this is a boolean Vector where
        every present value in this vector is set to True.

        """
        raise NotImplementedError

    @property
    def S(self):
        """Return the vector "structure".  This is the same as calling
        `Vector.pattern()` with no arguments.

        >>> v = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> assert v.S == v.pattern()

        """
        raise NotImplementedError

    def nonzero(self):
        """Select vector of nonzero entries."""
        pass

    def __setitem__(self, index, value):
        raise NotImplementedError

    def assign(self, value, index=None, mask=None, accum=None, desc=None):
        """Assign vector to vector.

        >>> v = Vector.sparse(types.INT8, 3)
        >>> w = Vector.from_1_to_n(3)
        >>> v[:] = w
        >>> print(v)
        0| 1
        1| 2
        2| 3

        If the index is another vector it is used as an assignment
        mask:

        >>> v.clear()
        >>> m = Vector.sparse(types.BOOL, 3)
        >>> m[1] = True
        >>> v[m] = w
        >>> print(v)
        0|
        1| 2
        2|
        >>> v.clear()
        >>> m = Vector.sparse(types.BOOL, 3)
        >>> m[1] = True
        >>> v[m] = 3
        >>> print(v)
        0|
        1| 3
        2|

        """
        pass

    def assign_scalar(self, value, index=None, mask=None, accum=None, desc=None):
        """Assign scalar to vector.

        >>> v = Vector.sparse(types.INT8, 3)
        >>> v[:] = 2
        >>> print(v)
        0| 2
        1| 2
        2| 2

        If the index is another vector it is used as an assignment
        mask:

        >>> v.clear()
        >>> m = Vector.sparse(types.BOOL, 3)
        >>> m[1] = True
        >>> v[m] = 3
        >>> print(v)
        0|
        1| 3
        2|

        """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __delitem__(self, index):
        raise NotImplementedError

    def extract_element(self, index):
        """Extract element from vector."""
        pass

    def extract(self, index, mask=None, accum=None, desc=None):
        """Extract subvector from vector."""
        pass

    def __contains__(self, index):
        raise NotImplementedError

    def get(self, i, default=None):
        """Get element at `i` or return `default` if not present.


        >>> M = Vector.from_lists([1, 2], [42, 149])
        >>> M.get(1)
        42
        >>> M.get(0) is None
        True
        >>> M.get(0, 'foo')
        'foo'

        """
        raise NotImplementedError

    def wait(self):
        """Wait for vector to complete."""
        pass

    def to_string(self, format_string="{:>%s}", width=2, prec=3, empty_char=""):
        """Return string representation of vector."""
        pass

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def print(self, level=2, name="A", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`..

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        pass
