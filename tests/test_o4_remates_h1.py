"""O4 remates h1: entrada_h1 y _bootstrap_cociente como cociente de sumas. Sin red."""
from __future__ import annotations

import numpy as np

from atisbo.exp import e3


def test_h1_entrada_h1_elige_005():
    por_alpha = [
        {"alpha": 0.01, "llamadas_llm": 111.0},
        {"alpha": 0.05, "llamadas_llm": 222.0},
        {"alpha": 0.10, "llamadas_llm": 333.0},
    ]
    ent = e3.entrada_h1(por_alpha)
    assert ent == por_alpha[1]
    assert float(ent["alpha"]) == 0.05
    assert float(ent["llamadas_llm"]) == 222.0
    sin_005 = [
        {"alpha": 0.01, "llamadas_llm": 111.0},
        {"alpha": 0.10, "llamadas_llm": 333.0},
    ]
    assert e3.entrada_h1(sin_005) is None


def test_h2_bootstrap_cociente_es_cociente_de_sumas():
    pares = [(1, 1), (0, 9), (2, 30), (0, 10)]
    res = e3._bootstrap_cociente(pares, n_remuestreos=1000, seed=0)
    esperado_media = 3.0 / 50.0
    assert abs(float(res["media"]) - esperado_media) < 1e-12
    n = len(pares)
    nums = np.asarray([float(p[0]) for p in pares], dtype=np.float64)
    dens = np.asarray([float(p[1]) for p in pares], dtype=np.float64)
    rng = np.random.RandomState(0)
    reps: list = []
    for _ in range(1000):
        idx = rng.randint(0, n, size=n)
        d = float(dens[idx].sum())
        reps.append(float(nums[idx].sum() / d) if d else 0.0)
    lo = float(np.percentile(reps, 2.5))
    hi = float(np.percentile(reps, 97.5))
    assert abs(float(res["ic95"][0]) - lo) < 1e-12
    assert abs(float(res["ic95"][1]) - hi) < 1e-12
    tasas = [float(m) / float(p) for m, p in pares]
    assert abs(sum(tasas) / len(tasas) - esperado_media) > 1e-9
    rng2 = np.random.RandomState(0)
    reps_medias: list = []
    for _ in range(1000):
        idx = rng2.randint(0, n, size=n)
        reps_medias.append(float(sum(tasas[i] for i in idx) / n))
    lo_m = float(np.percentile(reps_medias, 2.5))
    hi_m = float(np.percentile(reps_medias, 97.5))
    assert abs(float(res["ic95"][0]) - lo_m) > 1e-12 or abs(float(res["ic95"][1]) - hi_m) > 1e-12
