"""Perturbaciones del §1: mover, añadir, bloqueo. Azar solo vía random.Random."""

from __future__ import annotations

import random

from atisbo.env.rutina import elegir_accion
from atisbo.estado import H, N_MAX, TIPOS, W, Accion, EstadoSim

MOVER = "mover"
ANADIR = "anadir"
BLOQUEO = "bloqueo"
DURACION_BLOQUEO = 5

_DELTA = {
    Accion.ARRIBA: (0, -1),
    Accion.ABAJO: (0, 1),
    Accion.IZQ: (-1, 0),
    Accion.DER: (1, 0),
}


def celdas_vacias(estado: EstadoSim) -> list[tuple[int, int]]:
    """Celdas sin bot, sin objeto, sin cubo y sin bloqueo, ordenadas."""
    ocupadas = {estado.bot}
    for _id, _t, x, y, _e in estado.objetos:
        ocupadas.add((x, y))
    for c in estado.cubos.values():
        ocupadas.add(c)
    if estado.bloqueo is not None:
        ocupadas.add(estado.bloqueo)
    vacias = [(x, y) for y in range(H) for x in range(W) if (x, y) not in ocupadas]
    return vacias


def _nuevo_id(estado: EstadoSim) -> int:
    m = 0
    for o in estado.objetos:
        if o[0] > m:
            m = o[0]
    return m + 1


def _frente(estado: EstadoSim) -> tuple[int, int] | None:
    """Celda a la que la rutina movería al bot; None si no se mueve o sale del tablero."""
    a = elegir_accion(estado)
    if a not in _DELTA:
        return None
    dx, dy = _DELTA[a]
    bx, by = estado.bot
    nx, ny = bx + dx, by + dy
    if not (0 <= nx < W and 0 <= ny < H):
        return None
    return (nx, ny)


def aplicar_mover(estado: EstadoSim, rng: random.Random) -> tuple[EstadoSim, bool]:
    libres = [o for o in estado.objetos if not o[4]]
    vacias = celdas_vacias(estado)
    if not libres or not vacias:
        return estado, False
    _id, _t, _x, _y, _e = rng.choice(libres)
    nx, ny = rng.choice(vacias)
    objs = tuple(sorted(
        tuple((o[0], o[1], nx, ny, o[4]) if o[0] == _id else o for o in estado.objetos),
        key=lambda o: o[0],
    ))
    return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=objs, cubos=estado.cubos, bloqueo=estado.bloqueo, paso=estado.paso), True


def aplicar_anadir(estado: EstadoSim, rng: random.Random) -> tuple[EstadoSim, bool]:
    total = len(estado.objetos) + (1 if estado.carga is not None else 0)
    vacias = celdas_vacias(estado)
    if total >= N_MAX or not vacias:
        return estado, False
    tipo = rng.choice(list(TIPOS))
    nx, ny = rng.choice(vacias)
    nid = _nuevo_id(estado)
    objs = tuple(sorted(estado.objetos + ((nid, tipo, nx, ny, False),), key=lambda o: o[0]))
    return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=objs, cubos=estado.cubos, bloqueo=estado.bloqueo, paso=estado.paso), True


def aplicar_bloqueo(estado: EstadoSim) -> tuple[EstadoSim, bool]:
    """Bloquea la celda frente al bot; no aplica si la rutina no se mueve."""
    frente = _frente(estado)
    if frente is None:
        return estado, False
    return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=estado.objetos, cubos=estado.cubos, bloqueo=frente, paso=estado.paso), True


def tipos_disponibles(estado: EstadoSim) -> list[str]:
    """Tipos aplicables ahora; bloqueo solo si la rutina se mueve."""
    out = []
    if any(not o[4] for o in estado.objetos) and celdas_vacias(estado):
        out.append(MOVER)
    total = len(estado.objetos) + (1 if estado.carga is not None else 0)
    if total < N_MAX and celdas_vacias(estado):
        out.append(ANADIR)
    if _frente(estado) is not None:
        out.append(BLOQUEO)
    return sorted(out)
