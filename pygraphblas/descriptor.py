"""Contains all automatically generated Descriptors from CFFI.

"""
import contextvars
from .base import lib, ffi, _check

current_desc = contextvars.ContextVar("current_desc")


class Descriptor:
    """Wrapper class around GraphBLAS Descriptors.

    Descriptors "describe" the various options that can be used to
    control many aspects of graphblas operation.

    GraphBLAS Descriptors have a field and a value.  All of the common
    Descriptors necessary to use the GraphBLAS are available from the
    `pygraphblas.descriptor` module.

    Descriptors can be combined with the `&` operator.

    Descriptor | Description
    --- | ---
    `T0`      | Transpose First Argument
    `T1`      | Transpose Second Argument
    `T0T1`    | Transpose Both First and Second Argument
    `C`       | Complement Mask
    `CT1`     | C & T1
    `CT0`     | C & T0
    `CT0T1`   | T & T0 & T1
    `R`       | Replace Result
    `RT0`     | R & T0
    `RT1`     | R & T1
    `RT0T1`   | R & T0 & T1
    `RC`      | R & C
    `RCT0`    | R & C & T0
    `RCT1`    | R & C & T
    `RCT0T1`  | R & C & T0 & T1
    `S`       | Structural Mask
    `ST0`     | S & T0
    `ST1`     | S & T1
    `ST0T1`   | S & T0 & T1
    `RS`      | R & S
    `RST0`    | R & S & T0
    `RST1`    | R & S & T1
    `RST0T1`  | R & S & T0 & T1
    `RSC`     | R & S & C
    `RSCT0`   | R & S & C & T0
    `RSCT1`   | R & S & C & T1
    `RSCT0T1` |R & S & C & T0 & T1

    """

    __slots__ = ("field", "value", "_desc", "token", "name")

    def __init__(self, desc=None, name=None):
        raise NotImplementedError

    def get_desc(self):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *errors):
        raise NotImplementedError

    def __del__(self):
        raise NotImplementedError

    def __and__(self, other):
        raise NotImplementedError

    def __setitem__(self, field, value):
        raise NotImplementedError

    def __getitem__(self, field):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __contains__(self, other):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


Default = Descriptor(ffi.NULL, "Default")

T0 = Descriptor(lib.GrB_DESC_T0, name="T0")
T1 = Descriptor(lib.GrB_DESC_T1, name="T1")
T0T1 = Descriptor(lib.GrB_DESC_T0T1, name="T0T1")

C = Descriptor(lib.GrB_DESC_C, name="C")
CT0 = Descriptor(lib.GrB_DESC_CT0, name="CT0")
CT1 = Descriptor(lib.GrB_DESC_CT1, name="CT1")
CT0T1 = Descriptor(lib.GrB_DESC_CT0T1, name="CT0T1")

R = Descriptor(lib.GrB_DESC_R, name="R")
RT0 = Descriptor(lib.GrB_DESC_RT0, name="RT0")
RT1 = Descriptor(lib.GrB_DESC_RT1, name="RT1")
RT0T1 = Descriptor(lib.GrB_DESC_RT0T1, name="RT0T1")

RC = Descriptor(lib.GrB_DESC_RC, name="RC")
RCT0 = Descriptor(lib.GrB_DESC_RCT0, name="RCT0")
RCT1 = Descriptor(lib.GrB_DESC_RCT1, name="RCT1")
RCT0T1 = Descriptor(lib.GrB_DESC_RCT0T1, name="RCT0T1")

S = Descriptor(lib.GrB_DESC_S, name="S")
ST0 = Descriptor(lib.GrB_DESC_ST0, name="ST0")
ST1 = Descriptor(lib.GrB_DESC_ST1, name="ST1")
ST0T1 = Descriptor(lib.GrB_DESC_ST0T1, name="ST0T1")

RS = Descriptor(lib.GrB_DESC_RS, name="RS")
RST0 = Descriptor(lib.GrB_DESC_RST0, name="RST0")
RST1 = Descriptor(lib.GrB_DESC_RST1, name="RST1")
RST0T1 = Descriptor(lib.GrB_DESC_RST0T1, name="RST0T1")

RSC = Descriptor(lib.GrB_DESC_RSC, name="RSC")
RSCT0 = Descriptor(lib.GrB_DESC_RSCT0, name="RSCT0")
RSCT1 = Descriptor(lib.GrB_DESC_RSCT1, name="RSCT1")
RSCT0T1 = Descriptor(lib.GrB_DESC_RSCT0T1, name="RSCT0T1")

__all__ = [
    "Descriptor",
    "T1",
    "T0",
    "T0T1",
    "C",
    "CT1",
    "CT0",
    "CT0T1",
    "R",
    "RT0",
    "RT1",
    "RT0T1",
    "RC",
    "RCT0",
    "RCT1",
    "RCT0T1",
    "S",
    "ST1",
    "ST0",
    "ST0T1",
    "RS",
    "RST1",
    "RST0",
    "RST0T1",
    "RSC",
    "RSCT1",
    "RSCT0",
    "RSCT0T1",
]
