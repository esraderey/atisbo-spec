"""O4 primera parte: control conforme del riesgo (PLIEGO §7). Ancla roja."""

from __future__ import annotations

import numpy as np
import pytest

from atisbo.conformal import lambda_hat, riesgo, tau_desde_lambda


def test_a_umbral_07_y_alpha_minimo():
    rng_cal = np.random.default_rng(0)
    n = 500
    s_cal = rng_cal.random(n)
    malas_cal = (s_cal > 0.7).astype(np.int64)
    lam = lambda_hat(s_cal, malas_cal, 0.05)
    assert lam >= 0.7
    rng_test = np.random.default_rng(1)
    m = 5000
    s_test = rng_test.random(m)
    malas_test = (s_test > 0.7).astype(np.int64)
    assert riesgo(s_test, malas_test, lam) <= 0.06
    assert np.isneginf(lambda_hat(s_cal, malas_cal, 0.001))


def test_b_monotonia_alpha():
    rng = np.random.default_rng(2)
    s = rng.random(200)
    malas = (rng.random(200) < s).astype(np.int64)
    grid = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    lams = [lambda_hat(s, malas, a) for a in grid]
    for i in range(len(lams) - 1):
        assert lams[i] <= lams[i + 1]


def test_c_alpha_cero_y_borde_max():
    rng = np.random.default_rng(3)
    n = 50
    s = rng.random(n)
    malas = (rng.random(n) < 0.3).astype(np.int64)
    assert np.isneginf(lambda_hat(s, malas, 0.0))
    r_max = float(np.mean(malas))
    alpha_exact = (n / (n + 1)) * r_max + 1 / (n + 1)
    lam_exact = lambda_hat(s, malas, alpha_exact)
    assert lam_exact == float(np.max(s))
    lam_menor = lambda_hat(s, malas, alpha_exact - 1e-6)
    assert lam_menor < float(np.max(s))


def test_d_garantia_en_esperanza():
    rng = np.random.default_rng(20260707)
    reps = 200
    n = 300
    m = 2000
    suma_05 = 0.0
    suma_10 = 0.0
    for _ in range(reps):
        s_cal = rng.random(n)
        malas_cal = (rng.random(n) < s_cal).astype(np.int64)
        lam_05 = lambda_hat(s_cal, malas_cal, 0.05)
        lam_10 = lambda_hat(s_cal, malas_cal, 0.10)
        s_test = rng.random(m)
        malas_test = (rng.random(m) < s_test).astype(np.int64)
        suma_05 += riesgo(s_test, malas_test, lam_05)
        suma_10 += riesgo(s_test, malas_test, lam_10)
    assert suma_05 / reps <= 0.05 + 0.005
    assert suma_10 / reps <= 0.10 + 0.005


def test_e_determinismo_y_tamanos():
    rng = np.random.default_rng(4)
    s = rng.random(20)
    malas = (rng.random(20) < 0.5).astype(np.int64)
    assert lambda_hat(s, malas, 0.1) == lambda_hat(s, malas, 0.1)
    with pytest.raises(ValueError):
        lambda_hat(np.zeros(10), np.zeros(9, dtype=np.int64), 0.1)
    with pytest.raises(ValueError):
        lambda_hat(np.zeros(0), np.zeros(0, dtype=np.int64), 0.1)
    with pytest.raises(ValueError):
        lambda_hat(s, malas, -0.1)
    with pytest.raises(ValueError):
        lambda_hat(s, malas, 1.1)
    s_nan = s.copy()
    s_nan[0] = np.nan
    with pytest.raises(ValueError):
        lambda_hat(s_nan, malas, 0.1)
    malas_2 = malas.copy()
    malas_2[0] = 2
    with pytest.raises(ValueError):
        lambda_hat(s, malas_2, 0.1)


def test_formula_literal_n9_con_empate():
    s = np.array([0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    malas = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    alpha = 0.31
    n = 9
    distintos = np.unique(s)
    r_hat = np.array([float(np.sum(malas[s <= lam]) / n) for lam in distintos])
    punt = (n / (n + 1)) * r_hat + 1 / (n + 1)
    factibles = distintos[punt <= alpha]
    esperado = float(np.max(factibles))
    assert esperado == 0.5
    assert lambda_hat(s, malas, alpha) == esperado
    punt_sin_factor = r_hat + 1 / (n + 1)
    sup_sin_factor = float(np.max(distintos[punt_sin_factor <= alpha]))
    assert sup_sin_factor == 0.3
    assert sup_sin_factor != esperado
    punt_sin_mas1 = (n / (n + 1)) * r_hat
    sup_sin_mas1 = float(np.max(distintos[punt_sin_mas1 <= alpha]))
    assert sup_sin_mas1 == 0.7
    assert sup_sin_mas1 != esperado
    infimo = float(np.min(factibles))
    assert infimo == 0.1
    assert infimo != esperado
    assert riesgo(s, malas, 0.2) == pytest.approx(1 / 9)
    assert tau_desde_lambda(float("-inf")) == float("inf")


def test_g_riesgo_lambda_menos_infinito_es_cero():
    import numpy as np

    from atisbo.conformal import riesgo

    rng = np.random.default_rng(11)
    s = rng.random(25)
    malas = (rng.random(25) < 0.5).astype(np.int64)
    assert riesgo(s, malas, float("-inf")) == 0.0
