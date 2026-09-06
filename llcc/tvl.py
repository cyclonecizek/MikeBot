"""Three-valued (Kleene) logic.

The evaluator computes `permitted = NOT trigger OR exception` for every
requirement. Under Kleene semantics an unknown trigger and an unknown
exception both propagate to an unknown result, and anything that is not
definitely TRUE blocks the launch. That is the whole conservative-direction
policy, expressed once.
"""

from __future__ import annotations

from enum import Enum


class Tri(Enum):
    FALSE = 0
    UNKNOWN = 1
    TRUE = 2

    def __str__(self) -> str:
        return self.name.lower()


T = Tri.TRUE
F = Tri.FALSE
U = Tri.UNKNOWN


def tri(value: bool | None) -> Tri:
    """Lift an optional bool. None means 'could not be determined'."""
    if value is None:
        return U
    return T if value else F


def not_(a: Tri) -> Tri:
    if a is U:
        return U
    return F if a is T else T


def and_(*args: Tri) -> Tri:
    """FALSE dominates; otherwise UNKNOWN dominates."""
    if any(a is F for a in args):
        return F
    if any(a is U for a in args):
        return U
    return T


def or_(*args: Tri) -> Tri:
    """TRUE dominates; otherwise UNKNOWN dominates."""
    if any(a is T for a in args):
        return T
    if any(a is U for a in args):
        return U
    return F


def compare(lhs: float | None, op: str, rhs: float | None) -> Tri:
    """Comparison that yields UNKNOWN when either operand is unavailable."""
    if lhs is None or rhs is None:
        return U
    if op == "<":
        return tri(lhs < rhs)
    if op == "<=":
        return tri(lhs <= rhs)
    if op == ">":
        return tri(lhs > rhs)
    if op == ">=":
        return tri(lhs >= rhs)
    if op == "==":
        return tri(lhs == rhs)
    raise ValueError(f"unknown operator {op!r}")
