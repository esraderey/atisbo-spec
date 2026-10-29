"""O4 remates parte 1 (conformal): validacion extra y potencia de la correccion."""

from __future__ import annotations

import numpy as np
import pytest

from atisbo.conformal import lambda_hat, riesgo


def test_c1_lambda_hat_rechaza_inf_en_s():
    s_base = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float64)
    malas = np.array([0, 1, 0, 1], dtype=np.int64)
    s_posinf = s_base.copy()
    s_posinf[0] = float("inf")
    with pytest.raises(ValueError):
        lambda_hat(s_posinf, malas, 0.1)
    s_neginf = s_base.copy()
    s_neginf[1] = float("-inf")
    with pytest.raises(ValueError):
        lambda_hat(s_neginf, malas, 0.1)


def test_c2_lambda_hat_rechaza_mala_menos_uno():
    s = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float64)
    malas = np.array([0, -1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError):
        lambda_hat(s, malas, 0.1)


def test_c3_riesgo_tamanos_distintos():
    s = np.zeros(5, dtype=np.float64)
    malas_una = np.zeros(1, dtype=np.int64)
    with pytest.raises(ValueError):
        riesgo(s, malas_una, 0.5)


def test_c4_potencia_correccion_muestra_finita():
    # n = 20, s ~ Uniforme[0, 1], mala | s ~ Bernoulli(s).
    # El riesgo exacto de un umbral lam es r(lam) = lam**2 / 2 (0 con -inf).
    rng = np.random.default_rng(12345)
    n = 20
    reps = 4000
    suma_05 = 0.0
    suma_10 = 0.0
    suma_plugin_05 = 0.0
    for _ in range(reps):
        s_cal = rng.random(n)
        malas_cal = (rng.random(n) < s_cal).astype(np.int64)
        lam_05 = lambda_hat(s_cal, malas_cal, 0.05)
        lam_10 = lambda_hat(s_cal, malas_cal, 0.10)
        suma_05 += 0.0 if np.isneginf(lam_05) else lam_05**2 / 2.0
        suma_10 += 0.0 if np.isneginf(lam_10) else lam_10**2 / 2.0
        # Control negativo: plug-in sin correccion, calculado en el propio test
        # (mayor s_i con R_hat_n <= alpha, sin factor n/(n+1) ni +1/(n+1)).
        orden = np.argsort(s_cal, kind="stable")
        s_ord = s_cal[orden]
        malas_ord = malas_cal[orden]
        r_hat = np.cumsum(malas_ord, dtype=np.float64) / float(n)
        factibles = s_ord[r_hat <= 0.05]
        lam_plugin = float(np.max(factibles)) if factibles.size else float("-inf")
        suma_plugin_05 += 0.0 if np.isneginf(lam_plugin) else lam_plugin**2 / 2.0
    assert suma_05 / reps <= 0.05
    assert suma_10 / reps <= 0.10
    assert suma_plugin_05 / reps > 0.05
