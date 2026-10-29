"""Generador determinista de trazas (§1). Única CLI de generación. Sin IO salvo CLI."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

from atisbo.env.mundo import aplicar, es_exito
from atisbo.env.perturbaciones import (
    ANADIR,
    BLOQUEO,
    DURACION_BLOQUEO,
    MOVER,
    aplicar_anadir,
    aplicar_bloqueo,
    aplicar_mover,
    tipos_disponibles,
)
from atisbo.env.rutina import elegir_accion
from atisbo.estado import H, TIPOS, W, Accion, EstadoSim

CUBOS_FIJOS: dict[str, tuple[int, int]] = {"rojo": (0, 0), "verde": (11, 0), "azul": (0, 11)}
PRESUPUESTO = 160
FILTRO_INICIAL = 120


class Episodio(list):
    """Traza del §1: lista de (EstadoSim, Accion) con metadatos."""

    def __init__(self, pasos=(), perturbaciones=(), exito=False):
        super().__init__(pasos)
        self.perturbaciones = list(perturbaciones)
        self.exito = bool(exito)


def estado_inicial(rng: random.Random) -> EstadoSim:
    cubos = dict(CUBOS_FIJOS)
    bot = (6, 6)
    ocupadas: set[tuple[int, int]] = {bot} | set(cubos.values())
    n = rng.randint(3, 6)
    objs: list[tuple[int, str, int, int, bool]] = []
    for i in range(1, n + 1):
        tipo = rng.choice(list(TIPOS))
        vacias = [(x, y) for y in range(H) for x in range(W) if (x, y) not in ocupadas]
        nx, ny = rng.choice(vacias)
        ocupadas.add((nx, ny))
        objs.append((i, tipo, nx, ny, False))
    objs.sort(key=lambda o: o[0])
    return EstadoSim(bot=bot, carga=None, objetos=tuple(objs), cubos=cubos, bloqueo=None, paso=0)


def _sin_bloqueo(estado: EstadoSim) -> EstadoSim:
    if estado.bloqueo is None:
        return estado
    return EstadoSim(bot=estado.bot, carga=estado.carga, objetos=estado.objetos, cubos=estado.cubos, bloqueo=None, paso=estado.paso)


def _resuelve_sin_perturbaciones(estado: EstadoSim) -> bool:
    s = estado
    for _ in range(FILTRO_INICIAL):
        if es_exito(s):
            return True
        s = aplicar(s, elegir_accion(s))
    return es_exito(s)


def generar(seed: int, n_episodios: int, p_perturb: float = 0.0) -> list[Episodio]:
    rng_ini = random.Random(f"{seed}:ini")
    rng_pert = random.Random(f"{seed}:pert")
    episodios: list[Episodio] = []
    for _ in range(n_episodios):
        s = estado_inicial(rng_ini)
        while not _resuelve_sin_perturbaciones(s):
            s = estado_inicial(rng_ini)
        pasos: list[tuple[EstadoSim, Accion]] = []
        perts: list[tuple[int, str]] = []
        timer = 0
        for _ in range(PRESUPUESTO):
            if es_exito(s):
                break
            if timer == 0:
                s = _sin_bloqueo(s)
            if p_perturb > 0.0 and rng_pert.random() < p_perturb:
                disp = tipos_disponibles(s)
                if disp:
                    tipo = rng_pert.choice(disp)
                    if tipo == MOVER:
                        s2, ok = aplicar_mover(s, rng_pert)
                        if ok:
                            s = s2
                            perts.append((s.paso, MOVER))
                    elif tipo == ANADIR:
                        s2, ok = aplicar_anadir(s, rng_pert)
                        if ok:
                            s = s2
                            perts.append((s.paso, ANADIR))
                    else:
                        s2, ok = aplicar_bloqueo(s)
                        if ok:
                            s = s2
                            timer = DURACION_BLOQUEO
                            perts.append((s.paso, BLOQUEO))
            a = elegir_accion(s)
            s_next = aplicar(s, a)
            pasos.append((s, a))
            s = s_next
            if timer > 0:
                timer -= 1
                if timer == 0:
                    s = _sin_bloqueo(s)
        episodios.append(Episodio(pasos, perturbaciones=perts, exito=es_exito(s)))
    return episodios


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera trazas deterministas (§1).")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    eps = generar(a.seed, a.n, a.p)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(eps, f, protocol=4)


if __name__ == "__main__":
    from atisbo.env.trazas import main

    main()
