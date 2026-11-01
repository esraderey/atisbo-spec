"""EstadoSim, Accion y vector() — §2."""
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


def estado_desde_mascaras(mascaras_por_id: dict[int, np.ndarray], frame: np.ndarray) -> EstadoSim:
    """EstadoSim desde máscaras SAM (PLIEGO §2). Sin bot, bot=None."""
    from atisbo.env.render import BLOQUEO_COLOR, BOT_COLOR, CELL, COLORES
    from atisbo.env.trazas import CUBOS_FIJOS

    bot = None
    bot_lado = -1
    bloqueo = None
    parche = None
    vistos: set[tuple[str, int, int]] = set()
    for _oid, m in mascaras_por_id.items():
        mask = np.asarray(m, dtype=bool)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        lado = max(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)
        cx = int(xs.mean() // CELL)
        cy = int(ys.mean() // CELL)
        if not (0 <= cx < 12 and 0 <= cy < 12):
            continue
        pix = np.asarray(frame)[mask]
        col = pix.mean(axis=0).astype(float)
        mejor = None
        mejor_tipo = None
        mejor_d = float("inf")
        d_bot = float(np.linalg.norm(col - np.asarray(BOT_COLOR, dtype=float)))
        if d_bot < mejor_d:
            mejor_d = d_bot
            mejor = "bot"
        d_bl = float(np.linalg.norm(col - np.asarray(BLOQUEO_COLOR, dtype=float)))
        if d_bl < mejor_d:
            mejor_d = d_bl
            mejor = "bloqueo"
        for tipo, cref in COLORES.items():
            d = float(np.linalg.norm(col - np.asarray(cref, dtype=float)))
            if d < mejor_d:
                mejor_d = d
                mejor = "color"
                mejor_tipo = tipo
        if mejor == "bot" and mejor_d <= 60.0 and lado >= 22:
            if bot is None or lado > bot_lado:
                bot = (cx, cy)
                bot_lado = lado
        elif mejor == "bloqueo" and mejor_d <= 30.0:
            bloqueo = (cx, cy)
        elif mejor == "color" and mejor_d <= 60.0:
            if lado <= 12:
                if parche is None:
                    parche = (cx, cy)
            elif 13 <= lado <= 23:
                vistos.add((mejor_tipo, cx, cy))
    if bot is None and parche is not None:
        bot = parche
    if bot is None:
        carga = None
    else:
        bx, by = bot
        zona = np.asarray(frame)[by * CELL + 5:by * CELL + 14, bx * CELL + 5:bx * CELL + 14].reshape(-1, 3).mean(axis=0).astype(float)
        carga = None
        carga_d = float("inf")
        for tipo, cref in COLORES.items():
            d = float(np.linalg.norm(zona - np.asarray(cref, dtype=float)))
            if d < carga_d:
                carga_d = d
                carga = tipo
        if carga_d > 60.0:
            carga = None
    objs = sorted(vistos, key=lambda t: (t[2], t[1], t[0]))
    objetos = tuple((i + 1, tipo, x, y, CUBOS_FIJOS[tipo] == (x, y)) for i, (tipo, x, y) in enumerate(objs))
    return EstadoSim(bot=bot, carga=carga, objetos=objetos, cubos=dict(CUBOS_FIJOS), bloqueo=bloqueo, paso=0)
