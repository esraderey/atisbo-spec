"""Transición determinista del mundo 2D (§1). Puro, sin IO ni azar."""

from __future__ import annotations

from atisbo.estado import H, W, Accion, EstadoSim

_DELTA = {
    Accion.ARRIBA: (0, -1),
    Accion.ABAJO: (0, 1),
    Accion.IZQ: (-1, 0),
    Accion.DER: (1, 0),
}


def _nuevo_id(objetos) -> int:
    m = 0
    for o in objetos:
        if o[0] > m:
            m = o[0]
    return m + 1


def aplicar(estado: EstadoSim, accion: Accion) -> EstadoSim:
    """Un paso determinista. Moverse contra borde/bloqueo = quedarse."""
    bx, by = estado.bot
    paso = estado.paso + 1
    if accion in _DELTA:
        dx, dy = _DELTA[accion]
        nx, ny = bx + dx, by + dy
        if not (0 <= nx < W and 0 <= ny < H):
            nx, ny = bx, by
        elif estado.bloqueo is not None and (nx, ny) == estado.bloqueo:
            nx, ny = bx, by
        return EstadoSim(bot=(nx, ny), carga=estado.carga, objetos=estado.objetos,
                         cubos=estado.cubos, bloqueo=estado.bloqueo, paso=paso)
    if accion is Accion.TOMAR:
        if estado.carga is not None:
            return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=estado.objetos,
                             cubos=estado.cubos, bloqueo=estado.bloqueo, paso=paso)
        cands = [o for o in estado.objetos if (not o[4]) and o[2] == bx and o[3] == by]
        if not cands:
            return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=estado.objetos,
                             cubos=estado.cubos, bloqueo=estado.bloqueo, paso=paso)
        cands.sort(key=lambda o: o[0])
        _id, tipo, _x, _y, _e = cands[0]
        resto = tuple(o for o in estado.objetos if o[0] != _id)
        return EstadoSim(bot=estado.bot, carga=tipo, objetos=resto,
                         cubos=estado.cubos, bloqueo=estado.bloqueo, paso=paso)
    if accion is Accion.SOLTAR:
        if estado.carga is None:
            return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=estado.objetos,
                             cubos=estado.cubos, bloqueo=estado.bloqueo, paso=paso)
        tipo = estado.carga
        nid = _nuevo_id(estado.objetos)
        if estado.cubos.get(tipo) == estado.bot:
            objs = tuple(sorted(estado.objetos + ((nid, tipo, bx, by, True),), key=lambda o: o[0]))
        else:
            objs = tuple(sorted(estado.objetos + ((nid, tipo, bx, by, False),), key=lambda o: o[0]))
        return EstadoSim(bot=estado.bot, carga=None, objetos=objs,
                         cubos=estado.cubos, bloqueo=estado.bloqueo, paso=paso)
    return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=estado.objetos,
                     cubos=estado.cubos, bloqueo=estado.bloqueo, paso=paso)


def es_exito(estado: EstadoSim) -> bool:
    """Éxito cuando no hay carga y todos los objetos están en su cubo."""
    if estado.carga is not None:
        return False
    if not estado.objetos:
        return False
    return all(o[4] for o in estado.objetos)
