"""EstadoSim, Accion y vector() — §2 parcial (sin EstadoSam)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

W = 12
H = 12
N_MAX = 6
TIPOS = ("rojo", "verde", "azul")


class Accion(Enum):
    ARRIBA = 0
    ABAJO = 1
    IZQ = 2
    DER = 3
    TOMAR = 4
    SOLTAR = 5
    ESPERAR = 6


@dataclass(frozen=True)
class EstadoSim:
    bot: tuple[int, int]
    carga: str | None
    objetos: tuple[tuple[int, str, int, int, bool], ...]
    cubos: dict[str, tuple[int, int]] = field(hash=False)
    bloqueo: tuple[int, int] | None = None
    paso: int = 0


def vector(estado: EstadoSim) -> np.ndarray:
    """Vector float32 en [0, 1], D fijo con relleno hasta N_MAX."""
    parts: list[list[float]] = []
    bx, by = estado.bot
    parts.append([bx / (W - 1), by / (H - 1)])
    if estado.carga is None:
        parts.append([0.0, 0.0, 0.0, 1.0])
    else:
        parts.append([1.0 if estado.carga == t else 0.0 for t in TIPOS] + [0.0])
    objs = sorted(estado.objetos, key=lambda o: o[0])
    for i in range(N_MAX):
        if i < len(objs):
            _id, tipo, x, y, en_cubo = objs[i]
            tipo_oh = [1.0 if tipo == t else 0.0 for t in TIPOS]
            parts.append([1.0, x / (W - 1), y / (H - 1)] + tipo_oh + [1.0 if en_cubo else 0.0])
        else:
            parts.append([0.0] * 7)
    if estado.bloqueo is None:
        parts.append([0.0, 0.0, 0.0])
    else:
        lx, ly = estado.bloqueo
        parts.append([1.0, lx / (W - 1), ly / (H - 1)])
    flat = [v for p in parts for v in p]
    return np.asarray(flat, dtype=np.float32)
