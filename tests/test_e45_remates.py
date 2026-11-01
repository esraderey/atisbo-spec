"""Remates O6 E4/E5 antes de corrida de pago (revisor ciego + mutaciones). Sin red."""
from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

import atisbo.exp.matriz as matriz_mod
import atisbo.modelo as modelo_mod
from atisbo.control import PredictorRuido
from atisbo.exp import e2, e4, e5
from atisbo.exp.e2 import PRECIO_IN_MUSD, PRECIO_OUT_MUSD
from atisbo.exp.matriz import Celda, correr_matriz
from atisbo.meta_api import ClienteMeta, MetaAPIError


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _e1_repo() -> Path:
    return _repo_root() / "results" / "E1.json"


def _e3_repo() -> Path:
    return _repo_root() / "results" / "E3.json"


def _e2sam_repo() -> Path:
    return _repo_root() / "results" / "E2_sam.json"


def _e2_json() -> dict:
    return json.loads((_repo_root() / "results" / "E2.json").read_text(encoding="utf-8"))


class _FalsoEsperar:
    def __init__(self, pin=10, pout=5):
        self.pin = int(pin)
        self.pout = int(pout)
        self.llamadas = 0

    def post(self, url, cabeceras, cuerpo):
        self.llamadas += 1
        resp = {"choices": [{"message": {"content": '{"accion":"ESPERAR"}'}}], "usage": {"prompt_tokens": self.pin, "completion_tokens": self.pout}}
        return 200, {}, json.dumps(resp).encode("utf-8")


class _FalsoInvalido:
    def __init__(self):
        self.llamadas = 0

    def post(self, url, cabeceras, cuerpo):
        self.llamadas += 1
        resp = {"choices": [{"message": {"content": "esto no es json"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return 200, {}, json.dumps(resp).encode("utf-8")


class _Prohibido:
    def post(self, url, cabeceras, cuerpo):
        raise AssertionError("no debe tocar la red")


def _api_lleno(tmp_path, monkeypatch, out_name="E4.json"):
    monkeypatch.setenv("META_API_KEY", "clave-test-o6")
    cache = tmp_path / "cache"
    out = tmp_path / out_name
    falso = _FalsoEsperar()
    e4.main(["--rapido", "--modo", "api", "--cache", str(cache), "--out", str(out)], transporte=falso)
    datos = json.loads(out.read_text(encoding="utf-8"))
    return cache, out, datos


def _ultimo_cache(cache: Path) -> Path:
    fich = sorted(cache.glob("*.json"), key=lambda p: p.stat().st_mtime)
    assert len(fich) > 5
    return fich[-1]


def test_t1_e3_inexistente_no_entrena(tmp_path, monkeypatch):
    vistos = []
    orig = modelo_mod.Atisbo.entrenar

    def espia(episodios, seed=None, epocas=None, **kw):
        vistos.append(int(seed) if seed is not None else None)
        return orig(episodios, seed=seed, epocas=epocas, **kw)

    monkeypatch.setattr(modelo_mod.Atisbo, "entrenar", espia)
    out = tmp_path / "E4.json"
    with pytest.raises(FileNotFoundError):
        e4.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e3", str(tmp_path / "noexiste.json")])
    assert vistos == []


def test_t2_cachemiss_restaura_coste(tmp_path, monkeypatch):
    cache, _o1, _d1 = _api_lleno(tmp_path, monkeypatch, "lleno.json")
    _ultimo_cache(cache).unlink()
    monkeypatch.setenv("META_API_KEY", "clave-test-o6")
    out2 = tmp_path / "replay.json"
    e4.main(["--rapido", "--modo", "replay", "--cache", str(cache), "--out", str(out2)], transporte=_Prohibido())
    datos = json.loads(out2.read_text(encoding="utf-8"))
    llm = datos["llm"]
    llamadas = int(llm["llamadas"])
    assert llamadas > 0
    assert int(llm["peticiones"]) == llamadas
    assert int(llm["tokens_entrada"]) == llamadas * 10
    assert int(llm["tokens_salida"]) == llamadas * 5


def test_t3_celda_parcial_fuera_de_planas(tmp_path, monkeypatch):
    cache, _o1, _d1 = _api_lleno(tmp_path, monkeypatch, "lleno3.json")
    _ultimo_cache(cache).unlink()
    monkeypatch.setenv("META_API_KEY", "clave-test-o6")
    out2 = tmp_path / "replay3.json"
    e4.main(["--rapido", "--modo", "replay", "--cache", str(cache), "--out", str(out2)], transporte=_Prohibido())
    datos = json.loads(out2.read_text(encoding="utf-8"))
    parciales = [c for c in datos["celdas"] if int(c["n_episodios"]) > 0 and int(c["cache_miss"]) > 0]
    assert len(parciales) >= 1
    for celda in parciales:
        assert celda["parcial"] == "solo episodios en caché; no comparable con otras celdas"
        for clave in celda["metricas"]:
            assert f"{clave}@{celda['nombre']}" not in datos["metricas"]
    for celda in datos["celdas"]:
        ev = celda["seeds_evaluadas"]
        assert isinstance(ev, list) and len(ev) >= 1
        for _k, v in celda["metricas"].items():
            assert len(v["por_seed"]) == len(ev)


def test_t4_rama_atisbo_igual_que_e2(tmp_path, monkeypatch):
    def falso_entrenar(episodios, seed=None, epocas=None, **kw):
        return PredictorRuido(0.3, seed=int(seed) if seed is not None else 0)

    monkeypatch.setattr(modelo_mod.Atisbo, "entrenar", falso_entrenar)
    args = ["--seeds", "0,1", "--n", "2", "--n-trazas", "20", "--epocas", "1", "--modo", "simulada"]
    out_e4 = tmp_path / "E4.json"
    e4.main([*args, "--out", str(out_e4)])
    d4 = json.loads(out_e4.read_text(encoding="utf-8"))
    out_e2 = tmp_path / "E2.json"
    e2.main([*args, "--out", str(out_e2), "--e1", str(_e1_repo())])
    d2 = json.loads(out_e2.read_text(encoding="utf-8"))
    base = {c["nombre"]: c for c in d4["celdas"]}["base"]
    claves = ("llamadas_llm", "exito", "pasos", "racha_media", "tasa_malas_aceptadas", "llamadas_por_sorpresa", "malas_aceptadas")
    for clave in claves:
        m4 = base["metricas"][clave]
        m2 = d2["metricas"][clave]
        assert float(m4["media"]) == float(m2["media"])
        assert [float(v) for v in m4["ic95"]] == [float(v) for v in m2["ic95"]]
        assert [float(v) for v in m4["por_seed"]] == [float(v) for v in m2["por_seed"]]
    assert float(base["metricas"]["racha_media"]["media"]) > 0.0


def test_t5_procedencia_matriz(tmp_path, monkeypatch):
    orig_gen = matriz_mod.generar_trazas
    orig_split = matriz_mod.split_episodios
    orig_cal = matriz_mod.calibrar_epsilon
    orig_correr = matriz_mod.correr_episodio
    orig_pol = matriz_mod.PoliticaSimulada
    orig_entrenar = modelo_mod.Atisbo.entrenar
    gen_ll = []
    gen_rets = []

    def spy_gen(seed, n, p):
        gen_ll.append((int(seed), int(n), float(p)))
        r = orig_gen(seed, n, p)
        gen_rets.append(r)
        return r

    split_rets = []

    def spy_split(eps, seed, *a, **k):
        tr, te = orig_split(eps, seed, *a, **k)
        split_rets.append((eps, tr, te))
        return tr, te

    cal_ll = []
    eps_guardados = []

    def spy_cal(modelo, eps):
        cal_ll.append((modelo, eps))
        v = orig_cal(modelo, eps)
        eps_guardados.append(float(v))
        return v

    ent_ll = []

    def spy_entrenar(episodios, seed=None, epocas=None, **kw):
        ent_ll.append(episodios)
        return orig_entrenar(episodios, seed=seed, epocas=epocas, **kw)

    correr_ll = []

    def spy_correr(predictor, politica, mundo, tau=None, epsilon=None, registro=None, **kw):
        correr_ll.append({"predictor": predictor, "epsilon": float(epsilon)})
        return orig_correr(predictor, politica, mundo, tau=tau, epsilon=epsilon, registro=registro, **kw)

    pol_ll = []

    class SpyPol(orig_pol):
        def __init__(self, seed=0, *a, **k):
            pol_ll.append(int(seed))
            super().__init__(seed=seed, *a, **k)

    monkeypatch.setattr(matriz_mod, "generar_trazas", spy_gen)
    monkeypatch.setattr(matriz_mod, "split_episodios", spy_split)
    monkeypatch.setattr(matriz_mod, "calibrar_epsilon", spy_cal)
    monkeypatch.setattr(matriz_mod, "correr_episodio", spy_correr)
    monkeypatch.setattr(matriz_mod, "PoliticaSimulada", SpyPol)
    monkeypatch.setattr(modelo_mod.Atisbo, "entrenar", spy_entrenar)
    celdas = [Celda(nombre="base"), Celda(nombre="h1", horizonte=1)]
    correr_matriz(celdas, [0, 1], 1, 20, 1, "simulada", str(tmp_path / "cache"), 1500)
    assert gen_ll == [(0, 20, 0.02), (0, 20, 0.0), (1, 20, 0.02), (1, 20, 0.0)]
    assert split_rets[0][0] is gen_rets[0]
    assert split_rets[1][0] is gen_rets[1]
    assert ent_ll[0] is split_rets[0][1]
    assert cal_ll[0][1] is split_rets[1][2]
    assert ent_ll[1] is split_rets[2][1]
    assert cal_ll[1][1] is split_rets[3][2]
    assert len(eps_guardados) == 2
    modelos = [c[0] for c in cal_ll]
    assert len(correr_ll) == 4
    assert correr_ll[0]["predictor"] is modelos[0]
    assert correr_ll[0]["epsilon"] == eps_guardados[0]
    assert correr_ll[1]["predictor"] is modelos[0]
    assert correr_ll[1]["epsilon"] == eps_guardados[0]
    assert correr_ll[2]["predictor"] is modelos[1]
    assert correr_ll[2]["epsilon"] == eps_guardados[1]
    assert correr_ll[3]["predictor"] is modelos[1]
    assert correr_ll[3]["epsilon"] == eps_guardados[1]
    assert pol_ll == [0, 0, 1, 1]
def test_t6_defectos_y_celdas(tmp_path, monkeypatch):
    e2d = _e2_json()
    s4, n4, nt4, ep4, *_ = e4._parse_args([])
    s5, n5, nt5, ep5, *_ = e5._parse_args([])
    assert s4 == [0, 1, 2, 3, 4] == list(e2d["config"]["seeds"])
    assert (int(n4), int(nt4), int(ep4)) == (10, 200, 10)
    assert (int(n5), int(nt5), int(ep5)) == (10, 200, 10)
    assert int(e2d["config"]["n_episodios_por_seed"]) == 10
    assert int(e2d["config"]["n_trazas"]) == 200
    assert int(e2d["config"]["epocas"]) == 10

    def _stub_config():
        return {"seeds": [0], "n_episodios_por_seed": 2, "n_trazas": 20, "epocas": 2, "modo": "simulada", "max_llm_calls": 1500, "cache": "c", "semillas_mundo": [1000], "modelo": "muse-spark-1.3"}

    cap4 = {}

    def spy4(celdas, seeds, n, n_trazas, epocas, modo, cache, max_llm_calls, transporte=None):
        cap4["celdas"] = list(celdas)
        return {"config": _stub_config(), "metricas": {}, "celdas": [], "llm": {"modo": modo, "modelo": "m", "llamadas": 0, "peticiones": 0, "llamadas_red": 0, "tokens_entrada": 0, "tokens_salida": 0, "coste_estimado_usd": 0.0, "cache_miss": 0}, "truncada": False, "pasos_parcial": 0}

    monkeypatch.setattr(e4, "correr_matriz", spy4)
    e4.main(["--rapido", "--modo", "simulada", "--out", str(tmp_path / "E4.json")])
    por4 = {c.nombre: c for c in cap4["celdas"]}
    assert set(por4) == {"base", "sin_sorpresa", "horizonte_1"}
    for c in por4.values():
        assert float(c.p) == 0.02
        assert float(c.tau) == 0.5

    cap5 = {}

    def spy5(celdas, seeds, n, n_trazas, epocas, modo, cache, max_llm_calls, transporte=None):
        cap5["celdas"] = list(celdas)
        return {"config": _stub_config(), "metricas": {}, "celdas": [], "llm": {"modo": modo, "modelo": "m", "llamadas": 0, "peticiones": 0, "llamadas_red": 0, "tokens_entrada": 0, "tokens_salida": 0, "coste_estimado_usd": 0.0, "cache_miss": 0}, "truncada": False, "pasos_parcial": 0}

    monkeypatch.setattr(e5, "correr_matriz", spy5)
    e5.main(["--rapido", "--modo", "simulada", "--out", str(tmp_path / "E5.json")])
    por5 = {c.nombre: c for c in cap5["celdas"]}
    assert set(por5) == {"p0", "p002", "p005"}
    for c in por5.values():
        assert float(c.tau) == 0.5
        assert int(c.horizonte) == 3
        assert bool(c.sin_sorpresa) is False


def test_t7_truncado_mayor_que_un_episodio(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "clave-test-o6")
    celdas = [Celda(nombre="base", p=0.02, tau=1.0, sin_sorpresa=False, horizonte=3), Celda(nombre="sin_sorpresa", p=0.02, tau=1.0, sin_sorpresa=True, horizonte=3), Celda(nombre="horizonte_1", p=0.02, tau=1.0, sin_sorpresa=False, horizonte=1)]
    res = correr_matriz(celdas, [0], 2, 20, 2, "api", str(tmp_path / "cache"), 200, transporte=_FalsoEsperar())
    assert bool(res["truncada"]) is True
    assert int(res["pasos_parcial"]) > 0
    por = {c["nombre"]: c for c in res["celdas"]}
    assert int(por["base"]["n_episodios"]) == 1
    assert int(por["sin_sorpresa"]["n_episodios"]) == 0
    assert int(por["horizonte_1"]["n_episodios"]) == 0
    assert int(res["metricas"]["llamadas_llm@base"]["n"]) == 1


def test_t8_llm_es_suma(tmp_path, monkeypatch):
    _cache, _out, datos = _api_lleno(tmp_path, monkeypatch, "suma.json")
    llm = datos["llm"]
    llamadas = int(llm["llamadas"])
    assert llamadas > 0
    assert int(llm["peticiones"]) == llamadas
    assert int(llm["tokens_entrada"]) == llamadas * 10
    assert int(llm["tokens_salida"]) == llamadas * 5
    esperado = (int(llm["tokens_entrada"]) * PRECIO_IN_MUSD + int(llm["tokens_salida"]) * PRECIO_OUT_MUSD) / 1e6
    assert float(llm["coste_estimado_usd"]) == pytest.approx(esperado)


def test_t9_llm_invalidas(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "clave-test-o6")
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "api", "--cache", str(tmp_path / "cache"), "--out", str(out)], transporte=_FalsoInvalido())
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert int(datos["llm"]["llm_invalidas"]) > 0


def test_t10_seeds_repetidas(tmp_path):
    with pytest.raises(ValueError):
        e4.main(["--seeds", "0,0", "--rapido", "--modo", "simulada", "--out", str(tmp_path / "a.json")])
    with pytest.raises(ValueError):
        e5.main(["--seeds", "0,0", "--rapido", "--modo", "simulada", "--out", str(tmp_path / "b.json")])


def test_t11_meta_reintenta(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "clave-test")
    payload = {"model": "m", "messages": [{"role": "user", "content": "hola"}]}

    def _ok(texto='{"accion":"ESPERAR"}'):
        return 200, {}, json.dumps({"choices": [{"message": {"content": texto}}], "usage": {}}).encode("utf-8")

    class Seq:
        def __init__(self, pasos):
            self.pasos = list(pasos)
            self.n = 0

        def post(self, url, cabeceras, cuerpo):
            i = min(self.n, len(self.pasos) - 1)
            self.n += 1
            paso = self.pasos[i]
            if isinstance(paso, Exception):
                raise paso
            return paso

    cli = ClienteMeta(modo="api", cache_dir=tmp_path / "c1", transporte=Seq([urllib.error.URLError("caida"), _ok()]), dormir=lambda s: None)
    texto, _u, _l = cli.completar(dict(payload))
    assert json.loads(texto) == {"accion": "ESPERAR"}

    t2 = Seq([(504, {}, b"x")])
    cli2 = ClienteMeta(modo="api", cache_dir=tmp_path / "c2", transporte=t2, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli2.completar(dict(payload))
    assert int(t2.n) == 1

    t4 = Seq([ConnectionResetError("corte"), ConnectionResetError("corte"), ConnectionResetError("corte")])
    cli4 = ClienteMeta(modo="api", cache_dir=tmp_path / "c4", transporte=t4, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli4.completar(dict(payload))
    assert int(t4.n) == 3


def test_t12_referencias_con_procedencia(tmp_path):
    out = tmp_path / "E4.json"
    e4.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e3", str(_e3_repo()), "--e2-sam", str(_e2sam_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    e3d = json.loads(_e3_repo().read_text(encoding="utf-8"))
    e2s = json.loads(_e2sam_repo().read_text(encoding="utf-8"))
    assert datos["referencias"]["comparable"] is False
    conf = {float(x["alpha"]): x for x in datos["referencias"]["conforme"]}
    assert set(conf) == {float(x["alpha"]) for x in e3d["por_alpha"]}
    for ref in e3d["por_alpha"]:
        a = float(ref["alpha"])
        got = conf[a]
        assert str(got["commit"]) == str(e3d["commit"])
        assert str(got["modo"]) == str(e3d["llm"]["modo"])
        assert [int(v) for v in got["seeds_evaluadas"]] == [int(v) for v in e3d["config"]["seeds_evaluadas"]]
        assert int(got["n_episodios"]) == int(ref["n_episodios"])
        assert [float(v) for v in got["llamadas_llm_ic95"]] == [float(v) for v in e3d["metricas"][f"llamadas_llm@{a}"]["ic95"]]
        assert [float(v) for v in got["exito_ic95"]] == [float(v) for v in e3d["metricas"][f"exito@{a}"]["ic95"]]
    esam = datos["referencias"]["e_sam"]
    assert str(esam["commit"]) == str(e2s["commit"])
    assert str(esam["modo"]) == str(e2s["llm"]["modo"])
    assert [int(v) for v in esam["seeds_evaluadas"]] == [int(v) for v in e2s["config"]["seeds_evaluadas"]]
    assert [float(v) for v in esam["llamadas_llm_ic95"]] == [float(v) for v in e2s["metricas"]["llamadas_llm"]["ic95"]]
    assert [float(v) for v in esam["exito_ic95"]] == [float(v) for v in e2s["metricas"]["exito"]["ic95"]]
    out2 = tmp_path / "E4b.json"
    e4.main(["--seeds", "0", "--n", "1", "--n-trazas", "20", "--epocas", "1", "--modo", "simulada", "--out", str(out2), "--e3", str(_e3_repo()), "--e2-sam", str(_e2sam_repo())])
    d2 = json.loads(out2.read_text(encoding="utf-8"))
    assert d2["referencias"]["comparable"] is False
