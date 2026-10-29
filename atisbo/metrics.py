"""Métricas de acierto para E0 (PLIEGO §9)."""

from __future__ import annotations


def acierto_at_1(pred, true) -> float:
    """1.0 si la primera acción coincide."""
    return 1.0 if list(pred)[0] == list(true)[0] else 0.0


def acierto_at_3(pred, true) -> float:
    """1.0 si las 3 acciones coinciden."""
    p, t = list(pred), list(true)
    return 1.0 if len(p) == 3 and len(t) == 3 and all(x == y for x, y in zip(p, t)) else 0.0
