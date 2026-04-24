"""High level wrapper around GraphBLAS Matrices.

"""
import sys
import weakref
import operator
import random
from array import array
from pathlib import Path
from functools import partial
import numpy as np

from .base import (
    lib,
    ffi,
    NULL,
    NoValue,
    _check as _base_check,
    _error_codes,
    _build_range,
    _get_select_op,
    _get_bin_op,
    GxB_INDEX_MAX,
    GraphBLASException,
)

from . import types
from .vector import Vector
from .scalar import Scalar
from .semiring import current_semiring, Semiring
from .binaryop import current_accum, current_binop, Accum
from .monoid import Monoid, current_monoid
from .selectop import SelectOp
from . import descriptor
from .descriptor import Descriptor, T0, current_desc, S

__all__ = ["Matrix"]
__pdoc__ = {"Matrix.__init__": False}

import numba


def _check(obj, res):
    raise NotImplementedError


class Matrix:
    """GraphBLAS Sparse Matrix

    This is a high-level wrapper around the GrB_Matrix C type using
    the [cffi](https://cffi.readthedocs.io/en/latest/) library.

    A Matrix supports many possible operations according to the
    GraphBLAS API.  Many of those operations have overloaded
    operators.

    Operator | Description | Default
    --- | --- | ---
    A @    B | Matrix Matrix Multiplication | type default PLUS_TIMES semiring
    v @    A | Vector Matrix Multiplication | type default PLUS_TIMES semiring
    A @    v | Matrix Vector Multiplication | type default PLUS_TIMES semiring
    A @=   B | In-place Matrix Matrix Multiplication | type default PLUS_TIMES semiring
    v @=   A | In-place Vector Matrix Multiplication | type default PLUS_TIMES semiring
    A @=   v | In-place Matrix Vector Multiplication | type default PLUS_TIMES semiring
    A \\|  B | Matrix Union | type default SECOND combiner
    A \\|= B | In-place Matrix Union | type default SECOND combiner
    A &    B | Matrix Intersection | type default SECOND combiner
    A &=   B | In-place Matrix Intersection | type default SECOND combiner
    A +    B | Matrix Element-Wise Union | type default PLUS combiner
    A +=   B | In-place Matrix Element-Wise Union | type default PLUS combiner
    A -    B | Matrix Element-Wise Union | type default MINUS combiner
    A -=   B | In-place Matrix Element-Wise Union | type default MINUS combiner
    A *    B | Matrix Element-Wise Intersection | type default TIMES combiner
    A *=   B | In-place Matrix Element-Wise Intersection | type default TIMES combiner
    A /    B | Matrix Element-Wise Intersection | type default DIV combiner
    A /=   B | In-place Matrix Element-Wise Intersection | type default DIV combiner
    A ==   B | Compare Element-Wise Union | type default EQ operator
    A !=   B | Compare Element-Wise Union | type default NE operator
    A <    B | Compare Element-Wise Union | type default LT operator
    A >    B | Compare Element-Wise Union | type default GT operator
    A <=   B | Compare Element-Wise Union | type default LE operator
    A >=   B | Compare Element-Wise Union | type default GE operator

    Note that all the above operator syntax is mearly sugar over
    various combinations of calling `Matrix.mxm`, `Matrix.mxv`,
    `pygraphblas.Vector.vxm`, `Matrix.eadd`, and `Matrix.emult`.

    """

    __slots__ = ("_matrix", "type", "_funcs", "_keep_alives")

    def __init__(self, matrix, typ=None):
        raise NotImplementedError

    def __del__(self):
        raise NotImplementedError

    @classmethod
    def sparse(cls, typ, nrows=None, ncols=None, fill=None, mask=None):
        """Create an empty sparse Matrix from the given type.  The dimensions
        can be specified with `nrows` and `ncols`.  If no dimensions
        are specified, they default to `GxB_INDEX_MAX`.

        >>> m = Matrix.sparse(types.UINT8)
        >>> m.nrows == lib.GxB_INDEX_MAX
        True
        >>> m.ncols == lib.GxB_INDEX_MAX
        True
        >>> m.nvals == 0
        True

        Optional row and column dimension bounds can be provided to
        the method:

        >>> m = Matrix.sparse(types.UINT8, 10, 10)
        >>> m.nrows == 10
        True
        >>> m.ncols == 10
        True
        >>> m.nvals == 0
        True

        One of the Python types `(bool, int, float, complex)` can be
        passed instead.  They are turned into `(BOOL, INT64, FP64,
        FC64)` respectively:

        >>> Matrix.sparse(int)
        <Matrix(INT64, nvals: 0)>

        A sparse matrix can be "filled" with a starting value using
        the `fill` parameter and the `mask` parameter.  The `mask` is
        required otherwise `fill` is ignored.

        >>> mask = Matrix.sparse(types.BOOL)
        >>> mask[1,1] = True
        >>> list(Matrix.sparse(float, fill=3.14, mask=mask))
        [(1, 1, 3.14)]

        If `mask` is provided but no `fill`, the `type.default_zero`
        value is used:

        >>> list(Matrix.sparse(float, mask=mask))
        [(1, 1, 0.0)]

        """
        raise NotImplementedError

    @classmethod
    def dense(cls, typ, nrows=None, ncols=None, fill=None, sparsity=None):
        """Return a dense Matrix nrows by ncols.

        If `sparsity` is provided it is used for the sparsity of the
        new matrix See the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)
        for details.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M)
              0  1  2
          0|  0  0  0|  0
          1|  0  0  0|  1
          2|  0  0  0|  2
              0  1  2

        If a `fill` value is present, use that, otherwise use the
        `self.type.default_zero` attribute of the given type.

        >>> M = Matrix.dense(types.UINT8, 3, 3, fill=1)
        >>> print(M)
              0  1  2
          0|  1  1  1|  0
          1|  1  1  1|  1
          2|  1  1  1|  2
              0  1  2

        A dense matrix can be the maximum possible dimension, in which
        case it is an "iso" valued matrix.

        >>> M = Matrix.dense(types.UINT8)
        >>> M.nrows == lib.GxB_INDEX_MAX
        True
        >>> M[42,42]
        0

        """
        pass

    @classmethod
    def iso(cls, value, nrows=None, ncols=None):
        """Build a dense "iso" matrix from a scalar value.

        This is similar to `Matrix.dense` but infers the type of the
        new Matrix from the provided vbalue.

        >>> M = Matrix.iso(3)
        >>> assert M[42,42] == 3

        >>> M = Matrix.iso(3, 2, 2)
        >>> print(M)
              0  1
          0|  3  3|  0
          1|  3  3|  1
              0  1

        If you change an iso matrix, it is no longer stored as an iso
        object, and your matrix will grow in size.

        >>> M[1,1] = 2
        >>> print(M)
              0  1
          0|  3  3|  0
          1|  3  2|  1
              0  1

        """
        pass

    @classmethod
    def from_lists(cls, I, J, V=None, nrows=None, ncols=None, typ=None):
        """Create a new matrix from the given lists of row indices, column
        indices, and values.  If nrows or ncols are not provided, they
        are computed from the max values of the provides row and
        column indices lists.

        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
        >>> M = Matrix.from_lists(I, J)
        >>> print(M)
              0  1  2  3  4  5  6
          0|     t     t         |  0
          1|              t     t|  1
          2|                 t   |  2
          3|  t     t            |  3
          4|                 t   |  4
          5|        t            |  5
          6|        t  t  t      |  6
              0  1  2  3  4  5  6
        >>> from pygraphblas.gviz import draw_graph
        >>> draw_graph(M, filename='docs/imgs/Matrix_from_lists')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_from_lists.png](../imgs/Matrix_from_lists.png)

        If the third argument is a scalar value instead of a list, it
        is used to construct an "iso" Matrix where all values equal
        that scalar.

        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
        >>> V = True
        >>> M = Matrix.from_lists(I, J, V)
        >>> print(M)
              0  1  2  3  4  5  6
          0|     t     t         |  0
          1|              t     t|  1
          2|                 t   |  2
          3|  t     t            |  3
          4|                 t   |  4
          5|        t            |  5
          6|        t  t  t      |  6
              0  1  2  3  4  5  6

        """
        pass

    @classmethod
    def from_diag(cls, v, k=0, desc=None):
        """
        GxB_Matrix_diag constructs a matrix from a vector.  Let n be the length of
        the v vector, from GrB_Vector_size (&n, v).  If k = 0, then C is an n-by-n
        diagonal matrix with the entries from v along the main diagonal of C, with
        C(i,i) = v(i).  If k is nonzero, C is square with dimension n+abs(k).  If k
        is positive, it denotes diagonals above the main diagonal, with C(i,i+k) =
        v(i).  If k is negative, it denotes diagonals below the main diagonal of C,
        with C(i-k,i) = v(i).  This behavior is identical to the MATLAB statement
        C = diag(v,k), where v is a vector, except that GxB_Matrix_diag can also
        do typecasting.

        >>> v = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> print(Matrix.from_diag(v))
              0  1  2
          0|  1      |  0
          1|     2   |  1
          2|        3|  2
              0  1  2
        >>> print(Matrix.from_diag(v, 1))
              0  1  2  3
          0|     1      |  0
          1|        2   |  1
          2|           3|  2
          3|            |  3
              0  1  2  3
        >>> print(Matrix.from_diag(v, -1))
              0  1  2  3
          0|            |  0
          1|  1         |  1
          2|     2      |  2
          3|        3   |  3
              0  1  2  3
        """
        pass

    @classmethod
    def from_mm(cls, mm_file):
        """Create a new matrix by reading a Matrix Market file.

        >>> from pathlib import Path
        >>> M = Matrix.from_mm(Path('docs/test_mm.mm'))
        >>> print(M)
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|              2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 7   |  4
          5|        8            |  5
          6|        9 10 11      |  6
              0  1  2  3  4  5  6

        """
        pass

    @classmethod
    def from_tsv(cls, tsv_file, typ, nrows, ncols, **kwargs):
        """Create a new matrix by reading a tab separated value file.

        >>> M = Matrix.from_tsv(Path('docs/test_tsvfile.tsv'), types.INT32, 7, 7)
        >>> print(M)
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|              2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 7   |  4
          5|        8            |  5
          6|        9 10 11      |  6
              0  1  2  3  4  5  6

        """
        pass

    @classmethod
    def from_csv(
        cls, csv_file, typ, nrows, ncols, one_based=True, **reader_kwargs
    ):  # pragma: nocover
        """Create a new matrix by reading a comma separated value file.

        kwargs to this function are passed to the underlying
        `csv.Reader` object, so you can control various options like
        quoting and alternate delimiters that way.

        >>> M = Matrix.from_csv(Path('docs/test_tsvfile.tsv'), types.INT32, 7, 7, delimiter='\\t')
        >>> print(M)
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|              2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 7   |  4
          5|        8            |  5
          6|        9 10 11      |  6
              0  1  2  3  4  5  6

        """
        pass

    @classmethod
    def binread(cls, bin_file, opener=Path.open):  # pragma: nocover
        """Create a new matrix by reading a SuiteSparse specific binary file."""
        pass

    from_binfile = binread

    @classmethod
    def random(
        cls,
        typ,
        nvals,
        nrows=lib.GxB_INDEX_MAX,
        ncols=lib.GxB_INDEX_MAX,
        make_pattern=False,
        make_symmetric=False,
        make_skew_symmetric=False,
        make_hermitian=True,
        no_diagonal=False,
        seed=None,
    ):  # pragma: nocover
        """Create a new random Matrix of the given type, number of rows,
        columns and values.  Other flags set additional properties the
        matrix will hold.

        >>> from .gviz import draw_graph
        >>> M = Matrix.random(types.UINT8, 20, 5, 5,
        ...                   make_symmetric=True, no_diagonal=True, seed=42)
        >>> draw_graph(M, filename='../docs/imgs/Matrix_random')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_random.png](../imgs/Matrix_random.png)

        """
        pass

    @classmethod
    def identity(cls, typ, nrows, value=None):
        """Return a new square identity Matrix of nrows with diagonal set to
        one.

        If one is None, use the default `Type.default_one` value.

        >>> M = Matrix.identity(types.UINT8, 3, value=42)
        >>> print(M)
              0  1  2
          0| 42      |  0
          1|    42   |  1
          2|       42|  2
              0  1  2

        """
        pass

    @classmethod
    def ssget(cls, name_or_id=None, binary_cache_dir=None):  # pragma: nocover
        """Load a matrix from the [SuiteSparse Matrix Market](https://sparse.tamu.edu/).

        See [the ssgetpy
        library](https://github.com/drdarshan/ssgetpy) for search
        argument:

        >>> from pprint import pprint
        >>> from operator import itemgetter
        >>> pprint(sorted(list(Matrix.ssget('Newman/karate')), key=itemgetter(0)))
        [('karate.mtx', <Matrix(BOOL, shape: (34, 34), nvals: 156)>)]

        """
        pass

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Matrix.  This is
        only used if interacting with the low level API.

        >>> M = Matrix.sparse(types.INT8)
        >>> M.gb_type == lib.GrB_INT8
        True

        """
        pass

    @property
    def nrows(self):
        """Return the number of Matrix rows.

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.nrows
        3

        """
        pass

    @property
    def ncols(self):
        """Return the number of Matrix columns.

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.ncols
        3

        """
        pass

    @property
    def shape(self):
        """Numpy-like description of matrix shape as 2-tuple (nrows, ncols).

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.shape
        (3, 3)

        """
        pass

    @property
    def square(self):
        """True if Matrix is square, else False.

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.square
        True
        >>> M = Matrix.sparse(types.UINT8, 3, 4)
        >>> M.square
        False

        """
        pass

    @property
    def nvals(self):
        """Return the number of values stored in the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> M.nvals
        3

        """
        pass

    @property
    def memory_usage(self):
        """Returns the memory usage of the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> assert M.memory_usage > 0
        """
        pass

    @property
    def T(self):
        """Compute transpose of the Matrix.  See `Matrix.transpose`.

        Note: This property can be expensive, if you need the
        transpose more than once, consider storing this in a local
        variable.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> MT = M.T
        >>> MT.iseq(M.transpose())
        True

        """
        pass

    @property
    def M(self):
        """Return the structural "mask" pattern of this matrix.  See
        `pattern()`.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 142])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|142      |  2
              0  1  2
        >>> print(M.M)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        """
        pass

    def dup(self, clear=False):
        """Create an duplicate Matrix.

        If `clear` is true return an empty duplicate.

        >>> A = Matrix.sparse(types.UINT8)
        >>> A[1,1] = 42
        >>> B = A.dup()
        >>> B[1,1]
        42
        >>> B is not A
        True
        >>> C = A.dup(True)
        >>> assert not C

        """
        pass

    @property
    def hyper_switch(self):
        """Get the hyper_switch threshold. (See SuiteSparse User Guide)

        >>> A = Matrix.sparse(types.UINT8)
        >>> hs = A.hyper_switch
        >>> 0 < hs < 1
        True

        """
        pass

    @hyper_switch.setter
    def hyper_switch(self, switch):
        """Set the hyper_switch threshold. (See SuiteSparse User Guide)

        >>> A = Matrix.sparse(types.UINT8)
        >>> A.hyper_switch = 0.5
        >>> hs = A.hyper_switch
        >>> hs == 0.5
        True

        """
        pass

    @property
    def format(self):
        """Get Matrix format. (See SuiteSparse User Guide)

        >>> A = Matrix.sparse(types.UINT8)
        >>> A.format == lib.GxB_BY_ROW
        True

        """
        pass

    @format.setter
    def format(self, format):
        """Set Matrix format. (See SuiteSparse User Guide)

        >>> A = Matrix.sparse(types.UINT8)
        >>> A.format = lib.GxB_BY_COL
        >>> A.format == lib.GxB_BY_COL
        True

        """
        pass

    @property
    def sparsity(self):
        """Get Matrix sparsity control. (See SuiteSparse User Guide)

        >>> A = Matrix.sparse(types.UINT8)
        >>> A.sparsity == lib.GxB_AUTO_SPARSITY
        True

        """
        pass

    @sparsity.setter
    def sparsity(self, sparsity):
        """Set Matrix sparsity control. (See SuiteSparse User Guide)

        >>> A = Matrix.sparse(types.UINT8)
        >>> A.sparsity = lib.GxB_FULL + lib.GxB_BITMAP
        >>> A.sparsity == lib.GxB_FULL + lib.GxB_BITMAP

        """
        pass

    @property
    def sparsity_status(self):
        """Set Matrix sparsity status. (See SuiteSparse User Guide)

        >>> A = Matrix.sparse(types.UINT8)
        >>> A.sparsity_status in [1,2,4,8]
        True

        """
        pass

    def pattern(self, typ=types.BOOL, out=None):
        """Return the pattern of the matrix where every present value in this
        matrix is set to identity value for the provided type which
        defaults to BOOL.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 142])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|142      |  2
              0  1  2
        >>> P = M.pattern()
        >>> print(P)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        Pre-constructed matrix can be passed as the `out` parameter:

        >>> C = Matrix.dense(types.BOOL, 3, 3)
        >>> P = M.pattern(out=C)
        >>> print(C)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        """
        raise NotImplementedError

    @property
    def S(self):
        """Return the vector "structure".  This is the same as calling
        `Matrix.pattern()` with no arguments.

        >>> M = Matrix.from_lists([0, 1, 2], [0, 1, 2], [1, 2, 3])
        >>> assert M.S == M.pattern()

        """
        raise NotImplementedError

    def binwrite(self, filename, comments="", opener=Path.open):  # pragma: nocover
        """Write this matrix using custom SuiteSparse binary format."""
        pass

    to_binfile = binwrite

    def to_lists(self):
        """Extract the rows, columns and values of the Matrix as 3 lists.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> M.to_lists()
        [[0, 1, 2], [1, 2, 0], [42, 314, 4224]]

        """
        pass

    def clear(self):
        """Clear the matrix.  This does not change the size but removes all
        values.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> M.nvals == 3
        True
        >>> M.clear()
        >>> print(M)
              0  1  2
          0|         |  0
          1|         |  1
          2|         |  2
              0  1  2

        """
        pass

    def resize(self, nrows=GxB_INDEX_MAX, ncols=GxB_INDEX_MAX):
        """Resize the matrix.  If the dimensions decrease, entries that fall
        outside the resized matrix are deleted.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 149])
        >>> M.shape
        (3, 3)
        >>> M.resize(10, 10)
        >>> print(M)
              0  1  2  3  4  5  6  7  8  9
          0|    42                        |  0
          1|      314                     |  1
          2|149                           |  2
          3|                              |  3
          4|                              |  4
          5|                              |  5
          6|                              |  6
          7|                              |  7
          8|                              |  8
          9|                              |  9
              0  1  2  3  4  5  6  7  8  9

        """
        pass

    def transpose(self, cast=None, out=None, mask=None, accum=None, desc=None):
        """Return Transpose of this matrix.

        This function can serve multiple interesting purposes
        including typecasting.  See the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|149      |  2
              0  1  2

        >>> MT = M.transpose()
        >>> print(MT)
              0  1  2
          0|      149|  0
          1| 42      |  1
          2|   314   |  2
              0  1  2

        >>> MT = M.transpose(cast=types.BOOL, desc=descriptor.T0)
        >>> print(MT)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        >>> N = M.dup(True)
        >>> MT = M.transpose(desc=descriptor.T0, out=N)
        >>> print(MT)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|149      |  2
              0  1  2

        """
        raise NotImplementedError

    def cast(self, cast, out=None):
        """Cast this matrix to the provided type.  If out is not provided, a
        new matrix is of the cast type is created.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|149      |  2
              0  1  2
        >>> N = M.cast(types.FP32)
        >>> print(N.to_string(width=5, prec=4))
                  0    1    2
            0|      42.0     |  0
            1|          314.0|  1
            2|149.0          |  2
                  0    1    2

        >>> N = M.cast(types.FP64)
        >>> print(N.to_string(width=5, prec=4))
                  0    1    2
            0|      42.0     |  0
            1|          314.0|  1
            2|149.0          |  2
                  0    1    2

        >>> N = M.cast(types.INT64)
        >>> print(N.to_string(width=5, prec=4))
                  0    1    2
            0|        42     |  0
            1|            314|  1
            2|  149          |  2
                  0    1    2

        """
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
        """Element-wise addition with other matrix

        Element-wise addition takes the set union of the patterns of A
        and B and applies a binary operator for all entries that
        appear in the set intersection of the patterns of A and B.
        The default operators is the `PLUS` binary operator of the
        output type.

        The only difference between element-wise multiplication and
        addition is the pattern of the result, and what happens to
        entries outside the intersection. With multiplication the
        pattern of T is the intersection; with addition it is the set
        union. Entries outside the set intersection are dropped for
        multiplication, and kept for addition; in both cases the
        operator is only applied to those (and only those) entries in
        the intersection. Any binary operator can be used
        interchangeably for either operation.

        >>> from .gviz import draw_graph
        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
        >>> V = list(range(len(I)))
        >>> A = Matrix.from_lists(I, J, V, 7, 7)
        >>> draw_graph(A, filename='docs/imgs/Matrix_eadd_A')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_eadd_A.png](../imgs/Matrix_eadd_A.png)

        >>> B = Matrix.from_lists(
        ...    [0, 1, 4, 6],
        ...    [1, 3, 5, 5],
        ...    [9, 1, 4, 7], 7, 7)
        >>> draw_graph(B, filename='docs/imgs/Matrix_eadd_B')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_eadd_B.png](../imgs/Matrix_eadd_B.png)

        >>> draw_graph(A.eadd(B), filename='docs/imgs/Matrix_eadd_C')
        <graphviz.dot.Digraph object at ...>
        >>> print(A.eadd(B))
              0  1  2  3  4  5  6
          0|     9     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                11   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6

        ![Matrix_eadd_C.png](../imgs/Matrix_eadd_C.png)

        This can also be accomplished with the `+` operators:

        >>> print(A + B)
              0  1  2  3  4  5  6
          0|     9     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                11   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MIN:
        ...     print(A + B)
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 4   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6

        `eadd` is also called when a monoid is used by name as an
        attribute of an object:

        >>> print(A.min_monoid(B))
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 4   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6


        >>> print(A.eadd(B, A.type.min_plus))
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 4   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6

        The following operators default to use `eadd`:

        Operator | Description | Default
        --- | --- | ---
        A \\|  B | Matrix Union | type default SECOND combiner
        A \\|= B | In-place Matrix Union | type default SECOND combiner
        A +    B | Matrix Element-Wise Union | type default PLUS combiner
        A +=   B | In-place Matrix Element-Wise Union | type default PLUS combiner
        A -    B | Matrix Element-Wise Union | type default MINUS combiner
        A -=   B | In-place Matrix Element-Wise Union | type default MINUS combiner

        """
        pass

    union = eadd

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
        """Element-wise multiplication with other matrix.

        Element-wise multiplication applies a binary operator
        element-wise on two matrices A and B, for all entries that
        appear in the set intersection of the patterns of A and B.
        Other operators other than addition can be used.

        The pattern of the result of the element-wise multiplication
        is exactly this set intersection. Entries in A but not B, or
        visa versa, do not appear in the result.

        The only difference between element-wise multiplication and
        addition is the pattern of the result, and what happens to
        entries outside the intersection. With multiplication the
        pattern of T is the intersection; with addition it is the set
        union. Entries outside the set intersection are dropped for
        multiplication, and kept for addition; in both cases the
        operator is only applied to those (and only those) entries in
        the intersection. Any binary operator can be used
        interchangeably for either operation.

        >>> from .gviz import draw_graph
        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
        >>> V = list(range(len(I)))
        >>> A = Matrix.from_lists(I, J, V, 7, 7)
        >>> draw_graph(A, filename='docs/imgs/Matrix_emult_A')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_emult_A.png](../imgs/Matrix_emult_A.png)

        >>> B = Matrix.from_lists(
        ...    [0, 1, 1, 6, 6],
        ...    [1, 4, 6, 3, 5],
        ...    [9, 1, 4, 7, 11], 7, 7)
        >>> draw_graph(B, filename='docs/imgs/Matrix_emult_B')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_emult_B.png](../imgs/Matrix_emult_B.png)

        >>> draw_graph(A.emult(B), filename='docs/imgs/Matrix_emult_C')
        <graphviz.dot.Digraph object at ...>
        >>> print(A.emult(B))
              0  1  2  3  4  5  6
          0|     0               |  0
          1|              2    12|  1
          2|                     |  2
          3|                     |  3
          4|                     |  4
          5|                     |  5
          6|          70         |  6
              0  1  2  3  4  5  6

        ![Matrix_emult_C.png](../imgs/Matrix_emult_C.png)

        This can also be accomplished with the `+` operators:

        >>> print(A * B)
              0  1  2  3  4  5  6
          0|     0               |  0
          1|              2    12|  1
          2|                     |  2
          3|                     |  3
          4|                     |  4
          5|                     |  5
          6|          70         |  6
              0  1  2  3  4  5  6

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MIN:
        ...     print(A * B)
              0  1  2  3  4  5  6
          0|     0               |  0
          1|              1     3|  1
          2|                     |  2
          3|                     |  3
          4|                     |  4
          5|                     |  5
          6|           7         |  6
              0  1  2  3  4  5  6

        `emult` is also called when a binary operator is accessed by
        name as an attribute of an object:

        >>> print(A.min(B))
              0  1  2  3  4  5  6
          0|     0               |  0
          1|              1     3|  1
          2|                     |  2
          3|                     |  3
          4|                     |  4
          5|                     |  5
          6|           7         |  6
              0  1  2  3  4  5  6


        The following operators default to using `emult`:

        Operator | Description | Default
        --- | --- | ---
        A &    B | Matrix Intersection | type default SECOND combiner
        A &=   B | In-place Matrix Intersection | type default SECOND combiner
        A *    B | Matrix Element-Wise Intersection | type default TIMES combiner
        A *=   B | In-place Matrix Element-Wise Intersection | type default TIMES combiner
        A /    B | Matrix Element-Wise Intersection | type default DIV combiner
        A /=   B | In-place Matrix Element-Wise Intersection | type default DIV combiner

        """
        pass

    intersection = emult

    def all(self, other, op):
        """Do all elements in self compare True with op to other?

        >>> from . import INT64
        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> N = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> assert M.all(N, INT64.EQ)
        >>> assert not M.all(N, INT64.GT)

        """
        pass

    def iseq(self, other):
        """Compare two matrices for equality returning True or False.

        Not to be confused with `==` which will return a matrix of
        BOOL values comparing *elements* for equality.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> N = M.dup()
        >>> M.iseq(N)
        True
        >>> del N[0, 1]
        >>> M.iseq(N)
        False

        """
        pass

    def isne(self, other):
        """Compare two matrices for inequality.  See `Matrix.iseq`."""
        pass

    def __iter__(self):
        """Iterate over the (row, col, value) triples of the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> sorted(list(iter(M)))
        [(0, 1, 42), (1, 2, 314), (2, 0, 4224)]

        """
        raise NotImplementedError

    def to_arrays(self):
        """Convert Matrix to tuple of three dense
        [array](https:/docs.python.org/3/library/array.html) objects.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> M.to_arrays()
        (array('L', [0, 1, 2]), array('L', [1, 2, 0]), array('q', [42, 314, 4224]))

        """
        pass

    @property
    def rows(self):
        """An cffi cdata array of row indexes present in the matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> list(M.rows)
        [0, 1, 2]

        """
        pass

    @property
    def I(self):
        """Iterator over `Matrix.rows`.
        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> list(M.I)
        [0, 1, 2]

        """
        pass

    @property
    def npI(self):
        """numpy array from `Matrix.rows`.
        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> M.npI
        array([0, 1, 2], dtype=uint64)

        """
        pass

    @property
    def cols(self):
        """An cdata array of column indexes present in the matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> list(M.cols)
        [1, 2, 0]

        """
        pass

    @property
    def J(self):
        """Iterator over `Matrix.cols`.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> list(M.J)
        [1, 2, 0]

        """
        pass

    @property
    def npJ(self):
        """numpy array from `Matrix.cols`.
        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> M.npJ
        array([1, 2, 0], dtype=uint64)

        """
        pass

    @property
    def vals(self):
        """An cdata array of values present in the matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> list(M.vals)
        [42, 314, 4224]

        """
        pass

    @property
    def V(self):
        """Iterator over `Matrix.vals`.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> list(M.V)
        [42, 314, 4224]

        """
        pass

    @property
    def npV(self):
        """numpy array from `Matrix.vals`.
        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> M.npV
        array([  42,  314, 4224])

        """
        pass

    def __getattr__(self, name):
        """Look up operators as attributes for the given object."""
        raise NotImplementedError

    def __len__(self):
        """Return the number of elements in the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 4224])
        >>> len(M)
        3

        """
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

    def __pow__(self, exponent):
        raise NotImplementedError

    def kronpow(self, exponent):
        """Do "Kronecker Power" expansion.  This is useful for graph
        generation through expanding patterns.  And it draws pretty
        pictures.

        >>> from .gviz import draw_matrix
        >>> initiator = Matrix.from_lists([0, 0, 1], [0, 1, 1], [0.77, 0.88, 0.99])
        >>> initiator.kronpow(0).iseq(Matrix.identity(types.FP64, 2))
        True
        >>> initiator.kronpow(1).iseq(initiator)
        True
        >>> M = initiator.kronpow(3)
        >>> g = draw_matrix(M, scale=40,
        ...     filename='docs/imgs/Matrix_kronpow')

        ![Matrix_kronpow.png](../imgs/Matrix_kronpow.png)

        """
        pass

    def reduce_bool(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce matrix to a boolean.

        >>> M = Matrix.sparse(types.INT8)
        >>> M.reduce_bool()
        False
        >>> M[0,1] = True
        >>> M.reduce_bool()
        True

        >>> M.reduce_bool(types.BOOL.LOR_MONOID)
        True
        """
        pass

    def reduce_int(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce matrix to an integer.

        >>> M = Matrix.sparse(types.INT8)
        >>> M.reduce_int()
        0
        >>> M[0,1] = 42
        >>> M[0,2] = 42
        >>> M.reduce_int()
        84
        >>> M.reduce_int(types.INT8.MIN_MONOID)
        42

        """
        pass

    def reduce_float(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce matrix to an float.

        >>> M = Matrix.sparse(types.FP32)
        >>> M.reduce_float()
        0.0
        >>> M[0,1] = 42.0
        >>> M[0,2] = 42.0
        >>> M.reduce_float()
        84.0

        """
        pass

    def reduce(self, mon=None, accum=None, desc=None):
        """Do a scalar reduce based on this object's type:

        >>> M = Matrix.random(types.UINT8, 10, 3, 3, seed=42)
        >>> M.reduce()
        133

        >>> M = Matrix.random(types.FP32, 10, 3, 3, seed=42)
        >>> M.reduce()
        2.937098979949951

        >>> M = Matrix.random(types.UINT8, 10, 3, 3, seed=42)
        >>> M.reduce(M.type.min_monoid)
        13

        >>> M = Matrix.random(types.BOOL, 10, 3, 3, seed=42)
        >>> M.reduce()
        True

        """
        pass

    def reduce_vector(
        self, mon=None, out=None, cast=None, mask=None, accum=None, desc=None
    ):
        """Reduce matrix to a vector.

        >>> M = Matrix.sparse(types.FP32, 3, 3)
        >>> print(M.reduce_vector())
        0|
        1|
        2|
        >>> M[0,1] = 42.0
        >>> M[0,2] = 42.0
        >>> M[2,0] = -42.0
        >>> print(M.reduce_vector())
        0|84.0
        1|
        2|-42.0

        >>> print(M.reduce_vector(types.FP32.MIN_MONOID))
        0|42.0
        1|
        2|-42.0

        >>> v = Vector.sparse(types.FP32, M.nrows)
        >>> print(M.reduce_vector(out=v))
        0|84.0
        1|
        2|-42.0

        >>> M = Matrix.sparse(types.BOOL, 3, 3)
        >>> print(M.reduce_vector())
        0|
        1|
        2|

        >>> M[0,1] = True
        >>> M[0,2] = True
        >>> M[2,0] = True
        >>> print(M.reduce_vector())
        0| t
        1|
        2| t

        If there is no `out` parameter, the newly created result
        vector can also be cast to a different type:

        >>> print(M.reduce_vector(cast=types.UINT8))
        0| 2
        1|
        2| 1

        """
        pass

    def apply(self, op, out=None, mask=None, accum=None, desc=None):
        """Apply Unary op to matrix elements.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.apply(types.INT64.ABS))
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2

        >>> print(M.apply(types.INT64.ABS))
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2
        """
        raise NotImplementedError

    def apply_first(self, first, op, out=None, mask=None, accum=None, desc=None):
        """Apply a binary operator to the entries in a matrix, binding the
        first input to a scalar first.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.apply_first(1, types.INT64.PLUS))
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2
        >>> N = Matrix.sparse(M.type, M.nrows, M.ncols)
        >>> print(M.apply_first(1, types.INT64.PLUS, out=N))
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2

        `apply_first` is also used when a `Matrix` is used "on the
        right" for math operations like `+-*.` with a scalar:

        >>> print(1 + M)
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2

        """
        pass

    def apply_second(self, op, second, out=None, mask=None, accum=None, desc=None):
        """Apply a binary operator to the entries in a matrix, binding the
        second input to a scalar second.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.apply_second(types.INT64.PLUS, 1))
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2

        `apply_second` is also used when a `Matrix` is used "on the
        left" for math operations like `+-*.` with a scalar:

        >>> print(M + 1)
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2

        """
        pass

    def select(self, op, thunk=None, out=None, mask=None, accum=None, desc=None):
        """Select elements that match the given select operation condition.
        Can be a string mapping to following operators:

        Operator | Library Operation | Definition
        ---   | --- | ---
        `>`   | lib.GxB_GT_THUNK | Select greater than 'thunk'.
        `<`   | lib.GxB_LT_THUNK | Select less than 'thunk'.
        `>=`  | lib.GxB_GE_THUNK | Select greater than or equal to 'thunk'.
        `<=`  | lib.GxB_LE_THUNK | Select less than or equal to 'thunk'.
        `!=`  | lib.GxB_NE_THUNK | Select not equal to 'thunk'.
        `==`  | lib.GxB_EQ_THUNK | Select equal to 'thunk'.
        `>0`  | lib.GxB_GT_ZERO  | Select greater than zero.
        `<0`  | lib.GxB_LT_ZERO  | Select less than zero.
        `>=0` | lib.GxB_GE_ZERO  | Select greater than or equal to zero.
        `<=0` | lib.GxB_LE_ZERO  | Select less than or equal to zero.
        `!=0` | lib.GxB_NONZERO  | Select nonzero value.
        `==0` | lib.GxB_EQ_ZERO  | Select equal to zero.
        `max` | no equivalent    | Select max values.
        `min` | no equivalent    | Select min values.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.select('>', 0))
              0  1  2
          0|         |  0
          1|         |  1
          2|149      |  2
              0  1  2
        >>> print(M.select('>=', 0))
              0  1  2
          0|         |  0
          1|        0|  1
          2|149      |  2
              0  1  2
        >>> print(M.select('<', 0))
              0  1  2
          0|   -42   |  0
          1|         |  1
          2|         |  2
              0  1  2
        >>> N = M.dup(clear=True)
        >>> M.select('<', 0, out=N) is N
        True
        >>> print(N)
              0  1  2
          0|   -42   |  0
          1|         |  1
          2|         |  2
              0  1  2
        >>> N = M.dup(clear=True)
        >>> M.select('min', out=N) is N
        True
        >>> print(N)
              0  1  2
          0|   -42   |  0
          1|         |  1
          2|         |  2
              0  1  2
        >>> N = M.dup(clear=True)
        >>> M.select('max', out=N) is N
        True
        >>> print(N)
              0  1  2
          0|         |  0
          1|         |  1
          2|149      |  2
              0  1  2
        """
        pass

    def tril(self, offset=None):
        """Select the lower triangular Matrix.

        The diagonal `offset` can be used to select all below any
        diagonal rank, positive towars the upper right coner and
        negative toward the lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.tril())
              0  1  2
          0|  0      |  0
          1|  0  0   |  1
          2|  0  0  0|  2
              0  1  2
        >>> print(M.tril(1))
              0  1  2
          0|  0  0   |  0
          1|  0  0  0|  1
          2|  0  0  0|  2
              0  1  2
        >>> print(M.tril(-1))
              0  1  2
          0|         |  0
          1|  0      |  1
          2|  0  0   |  2
              0  1  2

        """
        pass

    def triu(self, offset=None):
        """Select the upper triangular Matrix.

        The diagonal `offset` can be used to select all above any
        diagonal rank, positive towars the upper right coner and
        negative toward the lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.triu())
              0  1  2
          0|  0  0  0|  0
          1|     0  0|  1
          2|        0|  2
              0  1  2
        >>> print(M.triu(1))
              0  1  2
          0|     0  0|  0
          1|        0|  1
          2|         |  2
              0  1  2
        >>> print(M.triu(-1))
              0  1  2
          0|  0  0  0|  0
          1|  0  0  0|  1
          2|     0  0|  2
              0  1  2

        """
        pass

    def diag(self, offset=None):
        """Select the diagonal Matrix.

        The diagonal `offset` can be used to select any diagonal rank,
        positive towars the upper right coner and negative toward the
        lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.diag())
              0  1  2
          0|  0      |  0
          1|     0   |  1
          2|        0|  2
              0  1  2
        >>> print(M.diag(1))
              0  1  2
          0|     0   |  0
          1|        0|  1
          2|         |  2
              0  1  2
        >>> print(M.diag(-1))
              0  1  2
          0|         |  0
          1|  0      |  1
          2|     0   |  2
              0  1  2

        """
        pass

    def vector_diag(self, k=0, desc=None):
        """
        GxB_Vector_diag extracts a vector v from an input matrix A, which
        may be rectangular.  If k = 0, the main diagonal of A is
        extracted; k > 0 denotes diagonals above the main diagonal of
        A, and k < 0 denotes diagonals below the main diagonal of A.
        Let A have dimension m-by-n.  If k is in the range 0 to n-1,
        then v has length min(m,n-k).  If k is negative and in the
        range -1 to -m+1, then v has length min(m+k,n).  If k is
        outside these ranges, v has length 0 (this is not an error).
        This function computes the same thing as the MATLAB statement
        v = diag(A,k) when A is a matrix, except that GxB_Vector_diag
        can also do typecasting.

        >>> from pygraphblas import UINT8
        >>> A = Matrix.dense(UINT8, 2, 2, fill=1)
        >>> print(A)
              0  1
          0|  1  1|  0
          1|  1  1|  1
              0  1
        >>> print(A.vector_diag())
        0| 1
        1| 1
        >>> print(A.vector_diag(1))
        0| 1
        >>> A.vector_diag(2)
        <Vector(UINT8 size: 0, nvals: 0)>
        >>> print(A.vector_diag(-1))
        0| 1
        >>> A.vector_diag(-2)
        <Vector(UINT8 size: 0, nvals: 0)>
        """
        pass

    def offdiag(self, offset=None):
        """Select the off-diagonal Matrix.

        The diagonal `offset` can be used to select off any diagonal
        rank, positive towars the upper right coner and negative
        toward the lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.offdiag())
              0  1  2
          0|     0  0|  0
          1|  0     0|  1
          2|  0  0   |  2
              0  1  2
        >>> print(M.offdiag(1))
              0  1  2
          0|  0     0|  0
          1|  0  0   |  1
          2|  0  0  0|  2
              0  1  2
        >>> print(M.offdiag(-1))
              0  1  2
          0|  0  0  0|  0
          1|     0  0|  1
          2|  0     0|  2
              0  1  2

        """
        pass

    def nonzero(self):
        """Select the non-zero Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M.nonzero())
              0  1  2
          0|    42   |  0
          1|         |  1
          2|149      |  2
              0  1  2

        """
        pass

    def _full(self):
        """"""
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

    def _get_args(self, mask=None, accum=None, desc=None):
        # if mask is not None and desc is None:
        #     desc = S
        raise NotImplementedError

    def mxm(
        self,
        other,
        semiring=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Matrix-matrix multiply.

        Multiply this matrix by `other` matrix.

        See Section 9.6 in the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)
        for details.

        `mxm` can be called directly or with the `@` operator:

        >>> m = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> n = Matrix.from_lists([0, 1, 2], [1, 2, 0], [2, 3, 4])
        >>> print(m)
              0  1  2
          0|     1   |  0
          1|        2|  1
          2|  3      |  2
              0  1  2
        >>> print(n)
              0  1  2
          0|     2   |  0
          1|        3|  1
          2|  4      |  2
              0  1  2

        Matrix multiply `m` by `n`:

        >>> o = m.mxm(n)
        >>> print(o)
              0  1  2
          0|        3|  0
          1|  8      |  1
          2|     6   |  2
              0  1  2

        Matrix matrix with the `@` operator:

        >>> o = m @ n
        >>> print(o)
              0  1  2
          0|        3|  0
          1|  8      |  1
          2|     6   |  2
              0  1  2

        By default, `mxm` and `@` create a new result matrix of the
        correct type and dimensions if one is not provided.  If you
        want to provide your own matrix to put the result in, you can
        pass it in the `out` parameter.  This is useful for
        accumulating results into a single matrix with minimal
        copying.  This is also supported by the `@=` syntax:

        >>> o = m.dup()
        >>> o.mxm(n, accum=types.INT64.min, out=o) is o
        True
        >>> print(o)
              0  1  2
          0|     1  3|  0
          1|  8     2|  1
          2|  3  6   |  2
              0  1  2
        >>> o = m.dup()
        >>> with Accum(types.INT64.min):
        ...     o @= n
        >>> print(o)
              0  1  2
          0|     1  3|  0
          1|  8     2|  1
          2|  3  6   |  2
              0  1  2

        The default semiring depends on the infered result type.  In
        the case of numbers, the default semiring is `PLUS_TIMES`.  In
        the case of type `BOOL`, it is `BOOL.LOR_LAND`.

        >>> from pygraphblas import INT64
        >>> o = m.mxm(n, semiring=INT64.min_plus)
        >>> print(o)
              0  1  2
          0|        4|  0
          1|  6      |  1
          2|     5   |  2
              0  1  2

        An explicit semiring can be passed to the method or provided
        with a context manager:

        >>> with INT64.min_plus:
        ...     o = m @ n
        >>> print(o)
              0  1  2
          0|        4|  0
          1|  6      |  1
          2|     5   |  2
              0  1  2

        Or the semiring can be accessed via an attribute on the
        matrix:

        >>> o = m.min_plus(n)
        >>> print(o)
              0  1  2
          0|        4|  0
          1|  6      |  1
          2|     5   |  2
              0  1  2


        Descriptors and accumulators can also be provided as an
        argument or a context manager:

        >>> descriptor.T0
        <Descriptor T0>
        >>> o = m.mxm(n, desc=descriptor.T0)
        >>> print(o)
              0  1  2
          0| 12      |  0
          1|     2   |  1
          2|        6|  2
              0  1  2
        >>> with descriptor.T0:
        ...     o = m @ n
        >>> print(o)
              0  1  2
          0| 12      |  0
          1|     2   |  1
          2|        6|  2
              0  1  2

        The accumulator context manager requires an extra `Accum`
        helper class to distinguish it from binary ops used in `eadd`
        and `emult`.

        Output can be cast to a specific type by passing the cast parameter:
        >>> o = m.mxm(n, cast=types.FP32)
        >>> print(o)
              0  1  2
          0|      3.0|  0
          1|8.0      |  1
          2|   6.0   |  2
              0  1  2
        """
        pass

    def mxv(
        self,
        other,
        semiring=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Matrix-vector multiply.

        Multiply this matrix by `other` column vector "on the right".
        For row vector multiplication "on the left" see `Vector.vxm`.

        See Section 9.6 in the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)
        for details.

        `mxv` can also be called directly or with the `@` operator:

        >>> from pygraphblas import INT64
        >>> m = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> v = Vector.from_lists([0, 1, 2], [2, 3, 4])
        >>> o = m.mxv(v)
        >>> print(o)
        0| 3
        1| 8
        2| 6
        >>> o = m @ v
        >>> print(o)
        0| 3
        1| 8
        2| 6

        By default, `mxv` and `@` create a new result matrix of the
        correct type and dimensions if one is not provided.  If you
        want to provide your own matrix to put the result in, you can
        pass it in the `out` parameter.  This is useful for
        accumulating results into a single matrix with minimal
        copying.

        >>> o = v.dup()
        >>> m.mxv(v, accum=INT64.plus, out=o) is o
        True
        >>> print(o)
        0| 5
        1|11
        2|10

        The default semiring depends on the infered result type.  In
        the case of numbers, the default semiring is `PLUS_TIMES`.  In
        the case of type `BOOL`, it is `BOOL.LOR_LAND`.

        An explicit semiring can be passed to the method or provided
        with a context manager:

        >>> o = m.mxv(v, semiring=INT64.min_plus)
        >>> print(o)
        0| 4
        1| 6
        2| 5

        >>> with INT64.min_plus:
        ...     o = m @ v
        >>> print(o)
        0| 4
        1| 6
        2| 5

        >>> o = m.min_plus(v)
        >>> print(o)
        0| 4
        1| 6
        2| 5

        Descriptors and accumulators can also be provided as an
        argument or a context manager:

        >>> o = m.mxv(v, desc=descriptor.T0)
        >>> print(o)
        0|12
        1| 2
        2| 6

        >>> with descriptor.T0:
        ...     o = m @ v
        >>> print(o)
        0|12
        1| 2
        2| 6

        >>> del o[1]
        >>> o = m.mxv(v, mask=o)
        >>> print(o)
        0| 3
        1|
        2| 6

        >>> o = m.mxv(v, cast=types.FP32)
        >>> print(o)
        0|3.0
        1|8.0
        2|6.0

        """
        pass

    def __matmul__(self, other):
        raise NotImplementedError

    def __imatmul__(self, other):
        raise NotImplementedError

    def kronecker(
        self, other, op=None, cast=None, out=None, mask=None, accum=None, desc=None
    ):
        """[Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).

        >>> n = Matrix.from_lists([0, 1, 2], [1, 2, 0], [2, 3, 4])
        >>> m = Matrix.dense(types.UINT64, 3, 3, fill=1)
        >>> print(n.kronecker(m))
              0  1  2  3  4  5  6  7  8
          0|           2  2  2         |  0
          1|           2  2  2         |  1
          2|           2  2  2         |  2
          3|                    3  3  3|  3
          4|                    3  3  3|  4
          5|                    3  3  3|  5
          6|  4  4  4                  |  6
          7|  4  4  4                  |  7
          8|  4  4  4                  |  8
              0  1  2  3  4  5  6  7  8

        >>> o = Matrix.sparse(types.UINT64, 9, 9)
        >>> m.kronecker(n, out=o) is o
        True
        >>> print(o)
              0  1  2  3  4  5  6  7  8
          0|     2        2        2   |  0
          1|        3        3        3|  1
          2|  4        4        4      |  2
          3|     2        2        2   |  3
          4|        3        3        3|  4
          5|  4        4        4      |  5
          6|     2        2        2   |  6
          7|        3        3        3|  7
          8|  4        4        4      |  8
              0  1  2  3  4  5  6  7  8

        >>> print(m.kronecker(n, op=types.UINT64.MIN))
              0  1  2  3  4  5  6  7  8
          0|     1        1        1   |  0
          1|        1        1        1|  1
          2|  1        1        1      |  2
          3|     1        1        1   |  3
          4|        1        1        1|  4
          5|  1        1        1      |  5
          6|     1        1        1   |  6
          7|        1        1        1|  7
          8|  1        1        1      |  8
              0  1  2  3  4  5  6  7  8
        """
        pass

    def extract_matrix(
        self,
        row_index=None,
        col_index=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Extract a submatrix.

        `GrB_Matrix_extract` extracts a submatrix from another matrix.
        The input matrix may be transposed first, via the descriptor.
        The result type remains the same.

        `row_index` and `col_index` can be slice objects that default
        to the equivalent of GrB_ALL.  Python slice objects support
        the SuiteSparse extensions for `GxB_RANGE`, `GxB_BACKWARDS`
        and `GxB_STRIDE`.  See the User Guide for details.

        The size of `C` is `|row_index|`-by-`|col_index|`.  Entries
        outside that sub-range are not accessed and do not take part
        in the computation.


        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M.extract_matrix())
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2

        >>> print(M.extract_matrix(0, 1))
              0
          0| 42|  0
              0

        >>> O = Matrix.sparse(types.UINT64, 1, 1)
        >>> M.extract_matrix(0, 1, out=O) is O
        True
        >>> print(O)
              0
          0| 42|  0
              0

        >>> print(M.extract_matrix(slice(1,2), 2))
              0
          0|  0|  0
          1|   |  1
              0

        >>> print(M.extract_matrix(0, slice(0,1)))
              0  1
          0|    42|  0
              0  1

        >>> N = Matrix.from_lists([1, 2], [2, 0], [True, True])
        >>> print(M[N])
              0  1  2
          0|         |  0
          1|        0|  1
          2|149      |  2
              0  1  2

        """
        pass

    def extract_col(
        self, col_index, row_slice=None, out=None, mask=None, accum=None, desc=None
    ):
        """Extract a column Vector.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2
        >>> print(M.extract_col(0))
        0|
        1|
        2|149

        >>> v = Vector.sparse(types.UINT64, M.ncols)
        >>> M.extract_col(0, out=v) is v
        True
        >>> print(v)
        0|
        1|
        2|149

        """
        pass

    def extract_row(
        self, row_index, col_slice=None, out=None, mask=None, accum=None, desc=None
    ):
        """Extract a row Vector.


        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2
        >>> print(M.extract_row(0))
        0|
        1|42
        2|

        """
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def assign_col(
        self, col_index, value, row_slice=None, mask=None, accum=None, desc=None
    ):
        """Assign a vector to a column.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)
        >>> M.assign_col(1, Vector.from_lists([1, 2], [True, True], 3))
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t   |  1
          2|     t   |  2
              0  1  2

        """
        pass

    def assign_row(
        self, row_index, value, col_slice=None, mask=None, accum=None, desc=None
    ):
        """Assign a vector to a row.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)
        >>> M.assign_row(1, Vector.from_lists([1, 2], [True, True], 3))
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t  t|  1
          2|         |  2
              0  1  2

        """
        pass

    def assign_matrix(
        self, value, rindex=None, cindex=None, mask=None, accum=None, desc=None
    ):
        """Assign a submatrix.

        Note: The name for this method `Matrix.assign_matrix()` is
        deprecated, use the name `Matrix.assign()` instead.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)
        >>> S = Matrix.sparse(types.BOOL, 3, 3)
        >>> S[1,1] = True
        >>> M.assign_matrix(S)
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t   |  1
          2|         |  2
              0  1  2

        >>> M.clear()

        Masked assignment with `M[key] = value` syntax can be done
        with if the index and value arguments are type `Matrix`:

        >>> M[S] = S
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t   |  1
          2|         |  2
              0  1  2

        """
        pass

    assign = assign_matrix

    def assign_scalar(
        self, value, row_slice=None, col_slice=None, mask=None, accum=None, desc=None
    ):
        """Assign a scalar `value` to the Matrix.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)

        The values of `row_slice` and `col_slice` determine what
        elements are assigned to the Matrix.  The value `None` maps to
        the GraphBLAS symbol `lib.GrB_ALL`, so the default behavior,
        with no other arguments, assigns the scalar to all elements:

        >>> M.assign_scalar(True)
        >>> print(M)
              0  1  2
          0|  t  t  t|  0
          1|  t  t  t|  1
          2|  t  t  t|  2
              0  1  2
        >>> M.clear()

        This is the same as the slice syntax with a bare colon:

        >>> M[:,:] = True
        >>> print(M)
              0  1  2
          0|  t  t  t|  0
          1|  t  t  t|  1
          2|  t  t  t|  2
              0  1  2
        >>> M.clear()

        If `row_slice` or `col_slice` is an integer, use it as an
        index to one row or column:

        >>> M.assign_scalar(True, 1)
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t  t|  1
          2|         |  2
              0  1  2
        >>> M.clear()

        An integer index and a scalar does row assignment:

        >>> M[1] = True
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t  t|  1
          2|         |  2
              0  1  2
        >>> M.clear()

        this is the same as the syntax:

        >>> M[1,:] = True
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t  t|  1
          2|         |  2
              0  1  2
        >>> M.clear()

        If `col_slice` is an integer, it does column assignment:

        >>> M.assign_scalar(True, None, 1)
        >>> print(M)
              0  1  2
          0|     t   |  0
          1|     t   |  1
          2|     t   |  2
              0  1  2
        >>> M.clear()

        Which is the same as the syntax:

        >>> M[:,1] = True
        >>> print(M)
              0  1  2
          0|     t   |  0
          1|     t   |  1
          2|     t   |  2
              0  1  2
        >>> M.clear()

        Just an integer index does a row assignment:

        >>> M.clear()
        >>> M[1] = Vector.from_lists([0,1], [True, True],3)
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t   |  1
          2|         |  2
              0  1  2
        >>> M.clear()

        >>> M[0:1,0:1] = True
        >>> print(M)
              0  1  2
          0|  t  t   |  0
          1|  t  t   |  1
          2|         |  2
              0  1  2
        >>> M.clear()

        """
        raise NotImplementedError

    def __setitem__(self, index, value):
        raise NotImplementedError

    def __delitem__(self, index):
        raise NotImplementedError

    def __contains__(self, index):
        raise NotImplementedError

    def get(self, i, j, default=None):
        """Get the element at row `i` col `j` or return the default value if
        the element is not present.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.get(1, 2)
        0
        >>> M.get(0, 0) is None
        True
        >>> M.get(0, 0, 'foo')
        'foo'

        """
        raise NotImplementedError

    def wait(self):
        """Wait for this Matrix to complete before allowing another thread to
        change it.

        """
        pass

    def to_markdown_table(self, title="A", width=2):
        """Return a string markdown table representation of the Matrix.

        >>> M = Matrix.from_lists([0, 0, 1, 2], [1, 2, 2, 0], [42, 2, 0, 149])
        >>> print(M.to_markdown_table())
        A|0|1|2
        ---|---|---|---
        0|   |42| 2
        1|   |  | 0
        2| 149|  |

        """
        pass

    def to_html_table(self, title="A", width=2):
        """Return a string markdown table representation of the Matrix.

        >>> M = Matrix.from_lists([0, 0, 1, 2], [1, 2, 2, 0], [42, 2, 0, 149])
        >>> print(M.to_html_table())
                <table>
                    <th>A</th>
                        <th>0</th>
                        <th>1</th>
                        <th>2</th>
        <BLANKLINE>
                    <tr>
                    <th>0</th>
                        <td>  </td>
                        <td>42</td>
                        <td> 2</td>
                    </tr>
        <BLANKLINE>
                    <tr>
                    <th>1</th>
                        <td>  </td>
                        <td>  </td>
                        <td> 0</td>
                    </tr>
        <BLANKLINE>
                    <tr>
                    <th>2</th>
                        <td>149</td>
                        <td>  </td>
                        <td>  </td>
                    </tr>
                </table>
        """
        pass

    def _repr_html_(self):  # pragma: nocover
        """jupyter notebook magic render method."""
        pass

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

    def to_string(
        self, format_string="{:>%s}", width=3, prec=5, empty_char="", cell_sep=""
    ):
        """Return a string representation of the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.to_string()
        '      0  1  2\\n  0|    42   |  0\\n  1|        0|  1\\n  2|149      |  2\\n      0  1  2'
        """
        pass

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    @classmethod
    def from_scipy_sparse(cls, m):
        """
        GrB_Type is inferred from m.dtype.

        >>> A = Matrix.from_lists([0, 1, 2], [1, 1, 2], [1, 2, 3])
        >>> s = A.to_scipy_sparse()
        >>> B = Matrix.from_scipy_sparse(s)
        >>> assert A.iseq(B)
        """
        pass

    def to_scipy_sparse(self, format="csr"):
        """Return a scipy sparse matrix of this Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.to_scipy_sparse()
        <3x3 sparse matrix of type '<class 'numpy.int64'>'...

        """
        pass

    def to_numpy(self):
        """Return a dense numpy matrix of this Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.to_numpy()
        array([[  0,  42,   0],
               [  0,   0,   0],
               [149,   0,   0]], dtype=int64)
        """
        pass

    def out_degree(self, typ=types.UINT64, out=None):
        """Return a UINT64 vector of the out-degree of this graph:

        >>> M = Matrix.from_lists([0, 1, 0, 2], [1, 2, 2, 0], [42, 0, 3, 149])
        >>> print(M.out_degree())
        0| 2
        1| 1
        2| 1

        """
        pass

    def gini(self, typ=types.FP64):
        """Calculate the Gini coefficient of the graph.

        >>> M = Matrix.random(types.UINT8, 10, 10, 10, seed=42)
        >>> M.gini()
        0.23333333333333334

        >>> M = Matrix.random(types.UINT8, 100, 10, 10, seed=42)
        >>> M.gini()
        0.0967741935483871

        >>> M = Matrix.random(types.UINT8, 10000, 100, 100, seed=42)
        >>> M.gini()
        0.0483808618504436

        >>> M = Matrix.dense(types.UINT8, 100, 100)
        >>> M.gini()
        0.0

        """
        pass
