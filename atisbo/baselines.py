"""Baselines de E0 (PLIEGO §4)."""

from __future__ import annotations

import random
from collections import Counter

from atisbo.estado import Accion

ORDEN = list(Accion)


def repetir_ultima(historia) -> tuple:
    """Repite la última acción 3 veces; historial vacío → ESPERAR."""
    h = list(historia)
    if not h:
        return (Accion.ESPERAR, Accion.ESPERAR, Accion.ESPERAR)
    return (h[-1], h[-1], h[-1])


class ngram3:
    """Frecuencias de la acción siguiente dadas las 2 últimas; desempate por orden de A."""

    def __init__(self) -> None:
        self._tri: dict[tuple, Counter] = {}
        self._uni: Counter = Counter()

    def ajustar(self, secuencias) -> "ngram3":
        self._tri = {}
        self._uni = Counter()
        for sec in secuencias:
            s = list(sec)
            for a in s:
                self._uni[a] += 1
            for i in range(2, len(s)):
                clave = (s[i - 2], s[i - 1])
                self._tri.setdefault(clave, Counter())[s[i]] += 1
        return self

    def _siguiente(self, par: tuple) -> Accion:
        if par in self._tri:
            c = self._tri[par]
            return min(c, key=lambda a: (-c[a], a.value))
        if self._uni:
            return min(self._uni, key=lambda a: (-self._uni[a], a.value))
        return Accion.ESPERAR

    def predecir(self, historia) -> tuple:
        h = list(historia)
        salida = []
        for _ in range(3):
            if len(h) >= 2:
                nxt = self._siguiente((h[-2], h[-1]))
            elif self._uni:
                nxt = min(self._uni, key=lambda a: (-self._uni[a], a.value))
            else:
                nxt = Accion.ESPERAR
            salida.append(nxt)
            h.append(nxt)
        return tuple(salida)


class azar:
    """Acciones uniformes con semilla fija."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def predecir(self, historia) -> tuple:
        return (
            self._rng.choice(ORDEN),
            self._rng.choice(ORDEN),
            self._rng.choice(ORDEN),
        )
