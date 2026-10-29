"""O4 remates E3 parte 2: procedencia, recálculo, coherencia, tau real, h1, truncado y duplicados. Sin red."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from atisbo.conformal import lambda_hat
from atisbo.exp import e3
from atisbo.exp.comun import split_episodios, validar, ventana


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema() -> dict:
    return json.loads((_repo_root() / "atisbo" / "exp" / "schema_results.json").read_text(encoding="utf-8"))


def _e1_repo() -> Path:
    return _repo_root() / "results" / "E1.json"


def _e1_media_llamadas() -> float:
    return float(json.loads(_e1_repo().read_text(encoding="utf-8"))["metricas"]["llamadas_llm"]["media"])


def test_t1_procedencia(tmp_path, monkeypatch):
    orig_gen = e3.generar_trazas
    orig_cal = e3.calibrar_epsilon
    orig_mundo = e3.Mundo
    orig_pol = e3.PoliticaSimulada
    orig_correr = e3.correr_episodio

    llamadas_gen: list = []
    ret_gen: list = []

    def spy_gen(seed, n_episodios, p_perturb):
        r = orig_gen(int(seed), int(n_episodios), float(p_perturb))
        llamadas_gen.append((int(seed), int(n_episodios), float(p_perturb)))
        ret_gen.append(r)
        return r

    monkeypatch.setattr(e3, "generar_trazas", spy_gen)

    llamadas_cal: list = []

    def spy_cal(modelo, episodios):
        llamadas_cal.append((modelo, list(episodios)))
        return orig_cal(modelo, episodios)

    monkeypatch.setattr(e3, "calibrar_epsilon", spy_cal)

    mundo_args: list = []
    mundos: list = []

    def spy_mundo(seed, p_perturb=0.0, *a, **k):
        mundo_args.append((int(seed), float(p_perturb)))
        m = orig_mundo(seed=int(seed), p_perturb=float(p_perturb), *a, **k)
        mundos.append(m)
        return m

    monkeypatch.setattr(e3, "Mundo", spy_mundo)

    pol_args: list = []
    politicas: list = []

    def spy_pol(seed=0, *a, **k):
        pol_args.append(int(seed))
        p = orig_pol(seed=int(seed), *a, **k)
        politicas.append(p)
        return p

    monkeypatch.setattr(e3, "PoliticaSimulada", spy_pol)

    corridas: list = []

    def spy_correr(predictor, politica, mundo, tau, epsilon, registro=None):
        corridas.append({"politica": politica, "mundo": mundo, "tau": float(tau), "epsilon": float(epsilon)})
        return orig_correr(predictor, politica, mundo, tau=float(tau), epsilon=float(epsilon), registro=registro)

    monkeypatch.setattr(e3, "correr_episodio", spy_correr)

    out = tmp_path / "E3.json"
    e3.main(["--rapido", "--modo", "simulada", "--alphas", "0.0,0.05", "--out", str(out), "--e1", str(_e1_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    s = 0
    assert llamadas_gen == [(s, 20, 0.02), (s, 20, 0.0), (2000 + s, 4, 0.02), (3000 + s, 4, 0.02)]
    assert len(llamadas_cal) == 1
    _, te_esperado = split_episodios(ret_gen[1], int(s))
    cal_eps = llamadas_cal[0][1]
    assert len(cal_eps) == len(te_esperado)
    for a, b in zip(cal_eps, te_esperado):
        assert len(a) == len(b)
        for (sa, aa), (sb, ab) in zip(a, b):
            assert aa == ab
            assert sa.bot == sb.bot
    assert mundo_args == [(1000 + s, 0.02), (1000 + s, 0.02)]
    assert pol_args == [int(s), int(s)]
    assert len(mundos) == 2 and mundos[0] is not mundos[1]
    assert len(politicas) == 2 and politicas[0] is not politicas[1]
    assert len(corridas) == 4
    assert corridas[0]["mundo"] is mundos[0]
    assert corridas[1]["mundo"] is mundos[0]
    assert corridas[2]["mundo"] is mundos[1]
    assert corridas[3]["mundo"] is mundos[1]
    assert corridas[0]["politica"] is politicas[0]
    assert corridas[2]["politica"] is politicas[1]
    ent0 = [x for x in datos["por_alpha"] if abs(float(x["alpha"]) - 0.0) < 1e-12][0]
    ent005 = [x for x in datos["por_alpha"] if abs(float(x["alpha"]) - 0.05) < 1e-12][0]
    tau0 = float("inf") if ent0["tau"][0] is None else float(ent0["tau"][0])
    tau005 = float("inf") if ent005["tau"][0] is None else float(ent005["tau"][0])
    assert math.isinf(corridas[0]["tau"]) and math.isinf(tau0)
    assert math.isinf(corridas[1]["tau"])
    assert corridas[2]["tau"] == tau005
    assert corridas[3]["tau"] == tau005
    eps = float(datos["config"]["epsilon_por_seed"][0])
    for c in corridas:
        assert c["epsilon"] == eps


def test_t2_recalculo_independiente(tmp_path, monkeypatch):
    from atisbo.modelo import Atisbo

    orig_train = Atisbo.entrenar
    guardado: dict = {}

    def spy_train(*a, **k):
        m = orig_train(*a, **k)
        guardado["modelo"] = m
        return m

    monkeypatch.setattr(Atisbo, "entrenar", spy_train)
    out = tmp_path / "E3.json"
    e3.main(["--rapido", "--modo", "simulada", "--alphas", "0.0,0.05", "--out", str(out), "--e1", str(_e1_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    modelo = guardado["modelo"]
    s = 0
    eps_cal = orig_gen_cal = __import__("atisbo.exp.comun", fromlist=["generar_trazas"]).generar_trazas(2000 + s, 4, 0.02)
    ss: list = []
    mm: list = []
    for ep in eps_cal:
        for t in range(len(ep)):
            pred = modelo.predecir(ventana(ep, t))
            ss.append(float(1.0 - float(pred.confianza)))
            mm.append(1 if pred.acciones[0] != ep[t][1] else 0)
    import numpy as np

    s_arr = np.asarray(ss, dtype=np.float64)
    m_arr = np.asarray(mm, dtype=np.int64)
    for ent in datos["por_alpha"]:
        af = float(ent["alpha"])
        lam_calc = float(lambda_hat(s_arr, m_arr, af))
        lam_json = ent["lambda"][0]
        if lam_json is None:
            assert math.isinf(lam_calc) and lam_calc < 0
        else:
            assert lam_calc == float(lam_json)
    eps_te = __import__("atisbo.exp.comun", fromlist=["generar_trazas"]).generar_trazas(3000 + s, 4, 0.02)
    for ent in datos["por_alpha"]:
        af = float(ent["alpha"])
        lam_json = ent["lambda"][0]
        lam = float("-inf") if lam_json is None else float(lam_json)
        num = 0
        den = 0
        for ep in eps_te:
            for t in range(len(ep)):
                pred = modelo.predecir(ventana(ep, t))
                sc = float(1.0 - float(pred.confianza))
                mala = 1 if pred.acciones[0] != ep[t][1] else 0
                den += 1
                if sc <= lam:
                    num += mala
        coc = float(num) / float(den) if den else 0.0
        assert abs(float(ent["riesgo_trazas"]) - coc) < 1e-12
        key = f"riesgo_trazas@{float(af)}"
        assert abs(float(datos["metricas"][key]["media"]) - coc) < 1e-12


def test_t3_coherencia_json(tmp_path):
    out = tmp_path / "E3.json"
    e3.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e1", str(_e1_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    for ent in datos["por_alpha"]:
        for lam, tau, acepta in zip(ent["lambda"], ent["tau"], ent["acepta_nada"]):
            assert bool(acepta) == (lam is None)
            if lam is None:
                assert tau is None
            else:
                assert abs(float(tau) - (1.0 - float(lam))) < 1e-12


def test_t4_lazo_usa_tau(tmp_path, monkeypatch):
    from atisbo.control import PredictorOraculo
    from atisbo.modelo import Atisbo

    monkeypatch.setattr(Atisbo, "entrenar", lambda *a, **k: PredictorOraculo())
    out = tmp_path / "E3.json"
    e3.main(["--rapido", "--modo", "simulada", "--p", "0.0", "--alphas", "0.0,0.5", "--out", str(out), "--e1", str(_e1_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    ent0 = [x for x in datos["por_alpha"] if abs(float(x["alpha"]) - 0.0) < 1e-12][0]
    ent05 = [x for x in datos["por_alpha"] if abs(float(x["alpha"]) - 0.5) < 1e-12][0]
    assert ent0["acepta_nada"] == [True]
    assert float(ent0["llamadas_llm"]) == float(ent0["pasos"])
    assert ent05["lambda"] == [0.0]
    assert ent05["tau"] == [1.0]
    assert float(ent05["llamadas_llm"]) == 0.0
    assert float(ent05["exito"]) == 1.0


def test_t5_h1_de_alpha_005(tmp_path):
    out = tmp_path / "E3.json"
    e3.main(["--rapido", "--modo", "simulada", "--alphas", "0.01,0.05,0.10", "--out", str(out), "--e1", str(_e1_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    ent005 = [x for x in datos["por_alpha"] if abs(float(x["alpha"]) - 0.05) < 1e-12][0]
    assert abs(float(datos["h1_detalle"]["ratio_llamadas"]) - float(ent005["llamadas_llm"]) / _e1_media_llamadas()) < 1e-12


def test_t6_truncado(tmp_path):
    out = tmp_path / "E3.json"
    cache_vacia = tmp_path / "cache_vacia"

    class _FallaSiUsa:
        def post(self, url, cabeceras, cuerpo):
            raise AssertionError("no debe tocar red en replay truncado")

    e3.main(
        ["--rapido", "--modo", "replay", "--max-llm-calls", "0", "--cache", str(cache_vacia),
         "--out", str(out), "--e1", str(_e1_repo())],
        transporte=_FallaSiUsa(),
    )
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    for ent in datos["por_alpha"]:
        assert int(ent["n_episodios"]) == 0
        assert ent["riesgo_empirico"] is None
        assert ent["tasa_condicional"] is None
        assert ent["llamadas_llm"] is None
        assert ent["exito"] is None
        assert ent["pasos"] is None
    assert int(datos["config"]["episodios_completados"]) == 0
    assert list(datos["config"]["seeds_evaluadas"]) == []


def test_t7_duplicados(tmp_path):
    out = tmp_path / "E3.json"
    with pytest.raises(ValueError):
        e3.main(["--rapido", "--modo", "simulada", "--alphas", "0.05,0.05", "--out", str(out), "--e1", str(_e1_repo())])
    with pytest.raises(ValueError):
        e3.main(["--rapido", "--modo", "simulada", "--seeds", "0,0", "--out", str(out), "--e1", str(_e1_repo())])
