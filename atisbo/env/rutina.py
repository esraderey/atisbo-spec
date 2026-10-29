"""Política de rutina (§1): define la acción buena."""

from __future__ import annotations

from atisbo.estado import Accion, EstadoSim


def _hacia(bot: tuple[int, int], destino: tuple[int, int]) -> Accion:
    dx = destino[0] - bot[0]
    dy = destino[1] - bot[1]
    if dx != 0:
        return Accion.DER if dx > 0 else Accion.IZQ
    if dy != 0:
        return Accion.ABAJO if dy > 0 else Accion.ARRIBA
    return Accion.ESPERAR


def elegir_accion(estado: EstadoSim) -> Accion:
    """Rutina: más cercano (Manhattan, desempate id menor) → cubo. Primero |dx| luego |dy|."""
    if estado.carga is not None:
        cubo = estado.cubos[estado.carga]
        if estado.bot == cubo:
            return Accion.SOLTAR
        return _hacia(estado.bot, cubo)
    libres = [o for o in estado.objetos if not o[4]]
    if not libres:
        return Accion.ESPERAR
    bx, by = estado.bot
    libres.sort(key=lambda o: (abs(o[2] - bx) + abs(o[3] - by), o[0]))
    _id, _tipo, ox, oy, _e = libres[0]
    if (bx, by) == (ox, oy):
        return Accion.TOMAR
    return _hacia(estado.bot, (ox, oy))
