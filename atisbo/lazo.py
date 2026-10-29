"""Lazo especulativo y mundo interactivo (PLIEGO 5; E1 variante LLM en cada paso)."""
from __future__ import annotations

import random

import numpy as np

from atisbo.env.mundo import aplicar, es_exito
from atisbo.env.perturbaciones import ANADIR, BLOQUEO, DURACION_BLOQUEO, MOVER, aplicar_anadir, aplicar_bloqueo, aplicar_mover, tipos_disponibles
from atisbo.env.rutina import elegir_accion
from atisbo.env.trazas import PRESUPUESTO, _resuelve_sin_perturbaciones, _sin_bloqueo, estado_inicial
from atisbo.estado import vector

K = 8


class Mundo:
    """Mundo interactivo que reproduce generar(seed, n, p) con la rutina como politica."""

    def __init__(self, seed: int, p_perturb: float = 0.0) -> None:
        self.seed = int(seed)
        self.p_perturb = float(p_perturb)
        self._rng_ini = random.Random(str(self.seed) + ":ini")
        self._rng_pert = random.Random(str(self.seed) + ":pert")
        self._timer = 0
        self._pasos = 0
        self.perturbaciones: list = []
        self.exito = False
        self._cur = None

    def _quizas_perturbar(self, s):
        if self.p_perturb > 0.0 and self._rng_pert.random() < self.p_perturb:
            disp = tipos_disponibles(s)
            if disp:
                tipo = self._rng_pert.choice(disp)
                if tipo == MOVER:
                    s2, ok = aplicar_mover(s, self._rng_pert)
                    if ok:
                        s = s2
                        self.perturbaciones.append((s.paso, MOVER))
                elif tipo == ANADIR:
                    s2, ok = aplicar_anadir(s, self._rng_pert)
                    if ok:
                        s = s2
                        self.perturbaciones.append((s.paso, ANADIR))
                else:
                    s2, ok = aplicar_bloqueo(s)
                    if ok:
                        s = s2
                        self._timer = DURACION_BLOQUEO
                        self.perturbaciones.append((s.paso, BLOQUEO))
        return s

    def reiniciar(self):
        s = estado_inicial(self._rng_ini)
        while not _resuelve_sin_perturbaciones(s):
            s = estado_inicial(self._rng_ini)
        self._timer = 0
        self._pasos = 0
        self.perturbaciones = []
        self.exito = False
        if self._timer == 0:
            s = _sin_bloqueo(s)
        s = self._quizas_perturbar(s)
        self._cur = s
        return s

    def avanzar(self, s, a):
        s_next = aplicar(s, a)
        self._pasos += 1
        if self._timer > 0:
            self._timer -= 1
            if self._timer == 0:
                s_next = _sin_bloqueo(s_next)
        if es_exito(s_next):
            self.exito = True
            self._cur = s_next
            return s_next
        if self._pasos >= PRESUPUESTO:
            self._cur = s_next
            return s_next
        if self._timer == 0:
            s_next = _sin_bloqueo(s_next)
        s_next = self._quizas_perturbar(s_next)
        self._cur = s_next
        return s_next


def _ventana_hist(historia) -> np.ndarray:
    if len(historia) >= K:
        seq = historia[-K:]
    else:
        seq = [historia[0]] * (K - len(historia)) + list(historia)
    return np.stack([vector(s) for s in seq]).astype(np.float32)


def correr_episodio(predictor, politica, mundo, tau, epsilon, registro=None):
    """Un episodio con el lazo 5 (solo variante LLM en cada paso)."""
    if registro is None:
        registro = []
    inv_antes = int(getattr(politica, "llm_invalidas", 0))
    pet_antes = int(getattr(politica, "peticiones", 0))
    red_antes = int(getattr(politica, "llamadas_red", 0))
    s = mundo.reiniciar()
    historia = [s]
    pasos = 0
    truncada = False
    while pasos < PRESUPUESTO and not es_exito(s):
        ventana = _ventana_hist(historia)
        pred = predictor.predecir(ventana)
        conf = float(pred.confianza)
        if conf < float(tau):
            accion, lat = politica.decidir(s)
            if bool(getattr(politica, "truncada", False)):
                truncada = True
                break
            rutina = elegir_accion(s)
            registro.append({"fuente": "llm", "conf": conf, "sorpresa": 0.0, "latencia_ms": float(lat), "accion": accion.name, "accion_rutina": rutina.name})
            s = mundo.avanzar(s, accion)
            historia.append(s)
            pasos += 1
            continue
        raise NotImplementedError("rama de Atisbo: O3")
    llamadas = sum(1 for e in registro if e.get("fuente") == "llm")
    latencias = [float(e.get("latencia_ms", 0.0)) for e in registro if e.get("fuente") == "llm"]
    rachas = []
    actual = 0
    for e in registro:
        if e.get("fuente") == "atisbo":
            actual += 1
        else:
            if actual > 0:
                rachas.append(actual)
                actual = 0
    if actual > 0:
        rachas.append(actual)
    racha_media = float(sum(rachas) / len(rachas)) if rachas else 0.0
    malas = sum(1 for e in registro if e.get("fuente") == "atisbo" and e.get("accion") != e.get("accion_rutina"))
    tasa = float(malas / pasos) if pasos else 0.0
    inv_ep = int(getattr(politica, "llm_invalidas", inv_antes)) - inv_antes
    pet_ep = int(getattr(politica, "peticiones", pet_antes)) - pet_antes
    red_ep = int(getattr(politica, "llamadas_red", red_antes)) - red_antes
    return {"pasos": int(pasos), "exito": bool(es_exito(s)), "llamadas_llm": int(llamadas), "peticiones": int(pet_ep), "llamadas_red": int(red_ep), "llm_invalidas": int(inv_ep), "truncada": bool(truncada or getattr(politica, "truncada", False)), "latencias_ms": latencias, "racha_media": float(racha_media), "malas_aceptadas": int(malas), "tasa_malas_aceptadas": float(tasa)}
