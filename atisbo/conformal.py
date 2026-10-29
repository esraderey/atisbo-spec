"""Control conforme del riesgo (PLIEGO §7): lambda_hat = sup{lambda : (n/(n+1)) * R_hat_n(lambda) + 1/(n+1) <= alpha} (Angelopoulos et al. 2022, Conformal Risk Control)."""

from __future__ import annotations

import numpy as np


def _validar(s: np.ndarray, malas: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    s_arr = np.asarray(s, dtype=np.float64)
    malas_arr = np.asarray(malas)
    if s_arr.size != malas_arr.size:
        raise ValueError("s y malas deben tener el mismo tamaño")
    if s_arr.size == 0:
        raise ValueError("s y malas no pueden estar vacíos")
    try:
        alpha_f = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha debe estar en [0, 1]")
    if not 0.0 <= alpha_f <= 1.0:
        raise ValueError("alpha debe estar en [0, 1]")
    if not np.all(np.isfinite(s_arr)):
        raise ValueError("s debe ser finito")
    if not np.all((malas_arr == 0) | (malas_arr == 1)):
        raise ValueError("malas debe estar en {0, 1}")
    return s_arr, malas_arr, alpha_f


def riesgo(s: np.ndarray, malas: np.ndarray, lam: float) -> float:
    """Media de 1[s <= lam] * malas; con lam = -inf da 0."""
    s_arr = np.asarray(s, dtype=np.float64).ravel()
    malas_arr = np.asarray(malas).ravel()
    if s_arr.size != malas_arr.size:
        raise ValueError("s y malas deben tener el mismo tamaño")
    lam_f = float(lam)
    if np.isneginf(lam_f):
        return 0.0
    aceptadas = s_arr <= lam_f
    return float(np.mean(aceptadas * malas_arr))


def lambda_hat(s: np.ndarray, malas: np.ndarray, alpha: float) -> float:
    """Mayor s_i distinto con (n/(n+1)) * R_hat_n + 1/(n+1) <= alpha; -inf si ninguno."""
    s_arr, malas_arr, alpha_f = _validar(s, malas, alpha)
    n = int(s_arr.size)
    distintos = np.unique(s_arr)
    mejor = float("-inf")
    for lam in distintos:
        lam_f = float(lam)
        r_hat = float(np.sum(malas_arr[s_arr <= lam_f]) / n)
        if (n / (n + 1)) * r_hat + 1 / (n + 1) <= alpha_f:
            mejor = lam_f
    return float(mejor)


def tau_desde_lambda(lam: float) -> float:
    """1 - lam, y +inf si lam es -inf."""
    lam_f = float(lam)
    if np.isneginf(lam_f):
        return float("inf")
    return 1.0 - lam_f
