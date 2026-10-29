"""Sorpresa L2 y calibración de épsilon (PLIEGO §5)."""
from __future__ import annotations

import numpy as np

from atisbo.estado import vector
from atisbo.exp.comun import ventana


def sorpresa(obs, pred_vec) -> float:
    """Norma L2 de vector(obs) − pred_vec; ValueError si las formas difieren."""
    v = np.asarray(vector(obs), dtype=np.float64)
    p = np.asarray(pred_vec, dtype=np.float64)
    if v.shape != p.shape:
        raise ValueError(f"formas distintas: {v.shape} frente a {p.shape}")
    return float(np.linalg.norm(v.ravel() - p.ravel()))


def errores_a_un_paso(predictor, episodios) -> list:
    """Para cada t con t+1: ‖vector(s_{t+1}) − predecir(ventana(ep,t)).estados[0]‖."""
    errores: list = []
    for ep in episodios:
        n = len(ep)
        if n < 2:
            continue
        for t in range(n - 1):
            vent = ventana(ep, t)
            pred = predictor.predecir(vent)
            errores.append(float(sorpresa(ep[t + 1][0], pred.estados[0])))
    return errores


def calibrar_epsilon(predictor, episodios) -> float:
    """Percentil 95 de los errores a un paso."""
    errores = errores_a_un_paso(predictor, episodios)
    if not errores:
        raise ValueError("sin errores: episodios demasiado cortos")
    return float(np.percentile(np.asarray(errores, dtype=np.float64), 95))
