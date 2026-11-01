"""Anclas E4/E5: ablaciones y robustez (PLIEGO §9 E4 y E5). Sin red."""
from __future__ import annotations

import json
import math
from pathlib import Path

import atisbo.exp.matriz as matriz_mod
import atisbo.modelo as modelo_mod
from atisbo.control import PredictorOraculo, PredictorRuido
from atisbo.exp import e2, e4, e5
from atisbo.exp.comun import validar
from atisbo.exp.matriz import Celda, correr_matriz
from atisbo.lazo import Mundo, correr_episodio
from atisbo.politica_llm import PoliticaSimulada


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema() -> dict:
    return json.loads((_repo_root() / "atisbo" / "exp" / "schema_results.json").read_text(encoding="utf-8"))


def _e3_repo() -> Path:
    return _repo_root() / "results" / "E3.json"


def _e2sam_repo() -> Path:
    return _repo_root() / "results" / "E2_sam.json"


def test_a_matriz_dos_celdas_y_e4_valida(tmp_path):
    celdas = [Celda(nombre="base"), Celda(nombre="h1", horizonte=1)]
    res = correr_matriz(celdas, [0], 2, 20, 1, "simulada", str(tmp_path / "cache"), 1500)
    assert set(res.keys()) >= {"config", "celdas", "metricas", "llm", "truncada", "pasos_parcial"}
    assert len(res["celdas"]) == 2
    for celda in res["celdas"]:
        assert "nombre" in celda
        assert "config" in celda
        assert "metricas" in celda
        for _k, v in celda["metricas"].items():
            assert "ic95" in v
            assert len(v["ic95"]) == 2
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "simulada", "--out", str(out)])
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    assert datos["experimento"] == "E4"
    assert len(datos["celdas"]) == 3
    por_nombre = {c["nombre"]: c for c in datos["celdas"]}
    assert set(por_nombre.keys()) == {"base", "sin_sorpresa", "horizonte_1"}
    for celda in datos["celdas"]:
        assert "config" in celda
        assert "metricas" in celda
        for _k, v in celda["metricas"].items():
            assert "ic95" in v


def test_b_e4_args_reales(tmp_path, monkeypatch):
    llamadas = []
    orig = matriz_mod.correr_episodio
    def espia(predictor, politica, mundo, tau, epsilon, registro=None, horizonte=3):
        llamadas.append({"tau": float(tau), "epsilon": float(epsilon), "horizonte": int(horizonte)})
        return orig(predictor, politica, mundo, tau=tau, epsilon=epsilon, registro=registro, horizonte=horizonte)
    monkeypatch.setattr(matriz_mod, "correr_episodio", espia)
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "simulada", "--out", str(out)])
    datos = json.loads(out.read_text(encoding="utf-8"))
    por_nombre = {c["nombre"]: c for c in datos["celdas"]}
    assert float(por_nombre["base"]["config"]["tau"]) == 0.5
    assert int(por_nombre["base"]["config"]["horizonte"]) == 3
    assert bool(por_nombre["sin_sorpresa"]["config"]["sin_sorpresa"]) is True
    assert int(por_nombre["horizonte_1"]["config"]["horizonte"]) == 1
    assert len(llamadas) == 6
    assert all(abs(c["tau"] - 0.5) < 1e-12 for c in llamadas)
    n_inf = sum(1 for c in llamadas if math.isinf(c["epsilon"]))
    assert n_inf == 2
    n_h1 = sum(1 for c in llamadas if c["horizonte"] == 1)
    assert n_h1 == 2
    n_base = sum(1 for c in llamadas if (not math.isinf(c["epsilon"])) and c["horizonte"] == 3)
    assert n_base == 2


def test_c_horizonte_llamadas():
    class Contador:
        def __init__(self):
            self.llamadas = 0
            self._o = PredictorOraculo()
        def predecir(self, ventana):
            self.llamadas += 1
            return self._o.predecir(ventana)
    c1 = Contador()
    m1 = Mundo(seed=0, p_perturb=0.0)
    p1 = PoliticaSimulada(seed=0)
    r1 = correr_episodio(c1, p1, m1, tau=0.5, epsilon=1e9, registro=[], horizonte=1)
    assert r1["pasos"] > 3
    assert c1.llamadas == int(r1["pasos"])
    assert int(r1["llamadas_llm"]) == 0
    c2 = Contador()
    m2 = Mundo(seed=0, p_perturb=0.0)
    p2 = PoliticaSimulada(seed=0)
    r2 = correr_episodio(c2, p2, m2, tau=0.5, epsilon=1e9, registro=[])
    assert int(r2["pasos"]) == int(r1["pasos"])
    assert int(c2.llamadas) == (int(r2["pasos"]) + 2) // 3
    assert int(c2.llamadas) < int(c1.llamadas)


def test_d_determinismo(tmp_path):
    o1 = tmp_path / "a.json"
    o2 = tmp_path / "b.json"
    e4.main(["--rapido", "--modo", "simulada", "--out", str(o1)])
    e4.main(["--rapido", "--modo", "simulada", "--out", str(o2)])
    d1 = json.loads(o1.read_text(encoding="utf-8"))
    d2 = json.loads(o2.read_text(encoding="utf-8"))
    d1.pop("fecha_utc")
    d2.pop("fecha_utc")
    assert d1 == d2


def test_e_referencias(tmp_path):
    out_miss = tmp_path / "miss.json"
    e4.main(["--rapido", "--modo", "simulada", "--out", str(out_miss), "--e2-sam", str(tmp_path / "noexiste.json")])
    d_miss = json.loads(out_miss.read_text(encoding="utf-8"))
    validar(d_miss, _schema())
    assert d_miss["referencias"]["e_sam"] == {"omitida": "sin O5"}
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e3", str(_e3_repo()), "--e2-sam", str(_e2sam_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    e3d = json.loads(_e3_repo().read_text(encoding="utf-8"))
    conf = datos["referencias"]["conforme"]
    assert len(conf) == 3
    for entrada, ref in zip(sorted(conf, key=lambda x: float(x["alpha"])), sorted(e3d["por_alpha"], key=lambda x: float(x["alpha"]))):
        assert float(entrada["alpha"]) == float(ref["alpha"])
        assert float(entrada["llamadas_llm"]) == float(ref["llamadas_llm"])
        assert float(entrada["exito"]) == float(ref["exito"])
        assert float(entrada["riesgo_empirico"]) == float(ref["riesgo_empirico"])
    e2sam = json.loads(_e2sam_repo().read_text(encoding="utf-8"))
    esam = datos["referencias"]["e_sam"]
    assert float(esam["llamadas_llm"]) == float(e2sam["metricas"]["llamadas_llm"]["media"])
    assert float(esam["exito"]) == float(e2sam["metricas"]["exito"]["media"])
    assert float(esam["coincidencia_todo"]) == float(e2sam["metricas"]["coincidencia_todo"]["media"])
    assert esam["comparacion_e2"] == e2sam["comparacion_e2"]


def test_f_e5_mundos(tmp_path, monkeypatch):
    vistos = []
    Real = matriz_mod.Mundo
    class Espia(Real):
        def __init__(self, seed=0, p_perturb=0.0):
            vistos.append({"seed": int(seed), "p": float(p_perturb)})
            super().__init__(seed=seed, p_perturb=p_perturb)
    monkeypatch.setattr(matriz_mod, "Mundo", Espia)
    out = tmp_path / "E5.json"
    e5.main(["--rapido", "--modo", "simulada", "--out", str(out)])
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    assert datos["experimento"] == "E5"
    por_nombre = {c["nombre"]: c for c in datos["celdas"]}
    assert float(por_nombre["p0"]["config"]["p"]) == 0.0
    assert float(por_nombre["p002"]["config"]["p"]) == 0.02
    assert float(por_nombre["p005"]["config"]["p"]) == 0.05
    assert len(vistos) == 3
    assert {v["p"] for v in vistos} == {0.0, 0.02, 0.05}
    assert {v["seed"] for v in vistos} == {1000}


def test_g_replay_sin_cache(tmp_path):
    cache_vacia = tmp_path / "cache_vacia"
    cache_vacia.mkdir()
    class FallaSiToca:
        def post(self, url, cabeceras, cuerpo):
            raise AssertionError("no debe tocar red en replay")
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "replay", "--cache", str(cache_vacia), "--out", str(out)], transporte=FallaSiToca())
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    total = 0
    for celda in datos["celdas"]:
        assert celda["metricas"] == {}
        assert celda.get("omitida") == "sin caché"
        assert int(celda["cache_miss"]) == 2
        assert int(celda["n_episodios"]) == 0
        total += int(celda["cache_miss"])
    assert int(datos["llm"]["cache_miss"]) == total
    assert total == 6


def test_h_truncada_api(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "clave-test")
    class Falso:
        def post(self, url, cabeceras, cuerpo):
            resp = {"choices": [{"message": {"content": '{"accion":"ESPERAR"}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            return 200, {}, json.dumps(resp).encode("utf-8")
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "api", "--max-llm-calls", "3", "--cache", str(tmp_path / "cache"), "--out", str(out)], transporte=Falso())
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    assert datos["truncada"] is True
    assert sum(int(c["n_episodios"]) for c in datos["celdas"]) == 0
    for celda in datos["celdas"]:
        assert celda["metricas"] == {}
    assert int(datos["pasos_parcial"]) > 0


def test_i_entrena_una_vez(tmp_path, monkeypatch):
    orig = modelo_mod.Atisbo.entrenar
    semillas = []
    def espia(episodios, seed, epocas):
        semillas.append(int(seed))
        return orig(episodios, seed=seed, epocas=epocas)
    monkeypatch.setattr(modelo_mod.Atisbo, "entrenar", espia)
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "simulada", "--out", str(out)])
    assert semillas == [0]


def test_j_horizonte_por_defecto():
    pred1 = PredictorRuido(0.3, seed=7)
    pol1 = PoliticaSimulada(seed=99)
    m1 = Mundo(seed=0, p_perturb=0.05)
    reg1: list = []
    r1 = correr_episodio(pred1, pol1, m1, tau=0.5, epsilon=0.01, registro=reg1)
    pred2 = PredictorRuido(0.3, seed=7)
    pol2 = PoliticaSimulada(seed=99)
    m2 = Mundo(seed=0, p_perturb=0.05)
    reg2: list = []
    r2 = correr_episodio(pred2, pol2, m2, tau=0.5, epsilon=0.01, registro=reg2, horizonte=3)
    assert r1 == r2
    assert reg1 == reg2


def test_k_base_es_e2(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    out_e2 = tmp_path / "E2.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out_e2), "--e1", str(e1_path)])
    datos_e2 = json.loads(out_e2.read_text(encoding="utf-8"))
    res = correr_matriz([Celda(nombre="base")], [0], 2, 20, 2, "simulada", str(tmp_path / "cache"), 1500)
    for clave in ("llamadas_llm", "exito", "pasos"):
        m_mat = res["metricas"][f"{clave}@base"]
        m_e2 = datos_e2["metricas"][clave]
        assert float(m_mat["media"]) == float(m_e2["media"])
        assert [float(v) for v in m_mat["por_seed"]] == [float(v) for v in m_e2["por_seed"]]
        assert [float(v) for v in m_mat["ic95"]] == [float(v) for v in m_e2["ic95"]]
