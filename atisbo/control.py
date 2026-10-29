"""Predictores de control (PLIEGO §3): misma API que Atisbo."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from atisbo.env.mundo import aplicar
from atisbo.env.rutina import elegir_accion
from atisbo.env.trazas import CUBOS_FIJOS
from atisbo.estado import H, N_MAX, TIPOS, W, Accion, vector


@dataclass
class Prediccion:
    acciones: tuple
    estados: np.ndarray
    confianza: float


def estado_desde_vector(vec) -> object:
    """Reconstruye el EstadoSim desde su vector (cubos fijos; ids por orden)."""
    from atisbo.estado import EstadoSim

    v = [float(x) for x in np.asarray(vec, dtype=np.float64).ravel()]
    bx = min(W - 1, max(0, int(round(v[0] * (W - 1)))))
    by = min(H - 1, max(0, int(round(v[1] * (H - 1)))))
    c = v[2:6]
    if c[3] >= 0.5:
        carga = None
    else:
        carga = TIPOS[int(np.argmax(c[:3]))]
    objs = []
    nid = 1
    base = 6
    for _ in range(N_MAX):
        pres, x, y, t0, t1, t2, enc = v[base:base + 7]
        base += 7
        if pres >= 0.5:
            ox = min(W - 1, max(0, int(round(x * (W - 1)))))
            oy = min(H - 1, max(0, int(round(y * (H - 1)))))
            tipo = TIPOS[int(np.argmax([t0, t1, t2]))]
            objs.append((nid, tipo, ox, oy, bool(enc >= 0.5)))
            nid += 1
    bpres, bx2, by2 = v[base:base + 3]
    if bpres >= 0.5:
        bloqueo = (
            min(W - 1, max(0, int(round(bx2 * (W - 1))))),
            min(H - 1, max(0, int(round(by2 * (H - 1))))),
        )
    else:
        bloqueo = None
    return EstadoSim(
        bot=(bx, by), carga=carga, objetos=tuple(objs),
        cubos=dict(CUBOS_FIJOS), bloqueo=bloqueo, paso=0,
    )


class PredictorOraculo:
    """Simula la rutina 3 pasos sin perturbaciones; conf = 1.0."""

    def predecir(self, ventana) -> Prediccion:
        s = estado_desde_vector(np.asarray(ventana)[-1])
        acciones = []
        estados = []
        for _ in range(3):
            a = elegir_accion(s)
            acciones.append(a)
            s = aplicar(s, a)
            estados.append(vector(s))
        return Prediccion(
            acciones=tuple(acciones),
            estados=np.stack(estados).astype(np.float32),
            confianza=1.0,
        )


class PredictorNulo:
    """Nunca especula; conf = 0.0."""

    def predecir(self, ventana) -> Prediccion:
        d = int(np.asarray(ventana).shape[1])
        return Prediccion(
            acciones=(Accion.ESPERAR, Accion.ESPERAR, Accion.ESPERAR),
            estados=np.zeros((3, d), dtype=np.float32),
            confianza=0.0,
        )


class PredictorRuido:
    """Rutina con prob. 1-p y acción uniforme con prob. p; conf = 1-p."""

    def __init__(self, p: float, seed: int = 0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p debe estar en [0, 1]")
        self._p = float(p)
        self._rng = random.Random(seed)

    def predecir(self, ventana) -> Prediccion:
        s = estado_desde_vector(np.asarray(ventana)[-1])
        acciones = []
        estados = []
        for _ in range(3):
            a_buena = elegir_accion(s)
            if self._rng.random() < self._p:
                a = self._rng.choice(list(Accion))
            else:
                a = a_buena
            acciones.append(a)
            s = aplicar(s, a)
            estados.append(vector(s))
        return Prediccion(
            acciones=tuple(acciones),
            estados=np.stack(estados).astype(np.float32),
            confianza=1.0 - self._p,
        )
