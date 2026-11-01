"""Anclas SAM-lazo parte 2 (PLIEGO §2, §6 y §9): el lazo viendo vía SAM. Sin red."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atisbo.control import PredictorNulo, PredictorOraculo
from atisbo.env.render import BLOQUEO_COLOR, BOT_COLOR, CELL, COLORES
from atisbo.env.trazas import CUBOS_FIJOS
from atisbo.estado import Accion
from atisbo.exp import e2
from atisbo.exp.comun import validar
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import CacheMiss, MetaAPIError
from atisbo.politica_llm import PoliticaSimulada
from atisbo.sam import ClienteSam, MundoSam, PredictorCiego


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _texto_real() -> str:
    for f in sorted((_repo_root() / "tests" / "fixtures" / "sam").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("payload", {}).get("medio") == "imagen":
            return str(d["texto"])
    raise AssertionError("sin fixture imagen")


def _schema() -> dict:
    return json.loads((_repo_root() / "atisbo" / "exp" / "schema_results.json").read_text(encoding="utf-8"))


def _mascaras_honestas(frame) -> dict[int, np.ndarray]:
    arr = np.asarray(frame, dtype=np.uint8)
    colores = {"__bot__": BOT_COLOR, "__bloqueo__": BLOQUEO_COLOR, **dict(COLORES)}
    out: dict[int, np.ndarray] = {}
    oid = 1
    for cy in range(12):
        for cx in range(12):
            y0, y1 = cy * CELL, (cy + 1) * CELL
            x0, x1 = cx * CELL, (cx + 1) * CELL
            celda = arr[y0:y1, x0:x1]
            for col in colores.values():
                m = np.all(celda == np.asarray(col, dtype=np.uint8), axis=2)
                if m.any():
                    full = np.zeros((384, 384), dtype=bool)
                    full[y0:y1, x0:x1] = m
                    out[oid] = full
                    oid += 1
    return out


def _mascaras_todo_en_cubo(frame) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    oid = 1
    for _tipo, (cx, cy) in CUBOS_FIJOS.items():
        full = np.zeros((384, 384), dtype=bool)
        full[cy * CELL:cy * CELL + 4, cx * CELL:cx * CELL + 17] = True
        out[oid] = full
        oid += 1
    return out


def _sse_cuerpo(chunks, uso=None):
    lineas = ["data: " + json.dumps({"type": "response.output_text.delta", "delta": c}) for c in chunks]
    lineas.append("data: " + json.dumps({"type": "response.completed", "response": {"usage": uso or {}}}))
    lineas.append("data: [DONE]")
    return ("\n".join(lineas) + "\n").encode("utf-8")


class _FakeTransporte:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append(url)
        idx = min(len(self.llamadas) - 1, len(self.respuestas) - 1)
        return self.respuestas[idx]


class _TransporteDoble:
    def __init__(self):
        self.llamadas = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append(url)
        if "chat/completions" in url:
            body = json.dumps({"choices": [{"message": {"content": '{"accion":"ESPERAR"}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode("utf-8")
            return (200, {}, body)
        if "/responses" in url:
            return (200, {}, _sse_cuerpo([], {"images_processed": 1}))
        raise AssertionError(url)


class _TransporteProhibido:
    def post(self, url, cabeceras, cuerpo):
        raise AssertionError("no debe tocar red en replay")


def test_a_lazo_percibido_termina_y_bot_1():
    mundo = MundoSam(Mundo(seed=1000, p_perturb=0.02), _mascaras_honestas)
    res = correr_episodio(PredictorNulo(), PoliticaSimulada(seed=0), mundo, tau=0.5, epsilon=0.1, registro=[])
    assert res["llamadas_llm"] == res["pasos"]
    assert mundo.coincidencias()["bot"] == 1.0


def test_b_real_sigue_dinamica_con_mentirosa():
    guion = [Accion.ARRIBA, Accion.DER, Accion.ABAJO, Accion.IZQ, Accion.TOMAR, Accion.SOLTAR, Accion.ESPERAR, Accion.ARRIBA, Accion.DER, Accion.ABAJO]
    env = MundoSam(Mundo(seed=123, p_perturb=0.0), lambda frame: {})
    perc = env.reiniciar()
    ref = Mundo(seed=123, p_perturb=0.0)
    s_ref = ref.reiniciar()
    assert env._real == s_ref
    assert perc.bot != s_ref.bot
    s_perc = perc
    for a in guion:
        s_perc = env.avanzar(s_perc, a)
        s_ref = ref.avanzar(s_ref, a)
        assert env._real == s_ref


def test_c_ciego_fuerza_llm():
    n = {"i": 0}

    def _ciega(frame):
        i = n["i"]
        n["i"] += 1
        if 3 <= i <= 5:
            return {}
        return _mascaras_honestas(frame)

    env = MundoSam(Mundo(seed=1000, p_perturb=0.0), _ciega)
    pred = PredictorCiego(PredictorOraculo(), env)
    registro: list = []
    correr_episodio(pred, PoliticaSimulada(seed=0), env, tau=0.5, epsilon=1e9, registro=registro)
    assert env.pasos_ciegos == 3
    assert len(registro) > 3
    assert registro[3].get("fuente") == "llm"


def test_d_exito_percibido_no_es_real(tmp_path):
    env = MundoSam(Mundo(seed=1000, p_perturb=0.0), _mascaras_todo_en_cubo)
    registro: list = []
    res = correr_episodio(PredictorNulo(), PoliticaSimulada(seed=0), env, tau=0.5, epsilon=0.1, registro=registro)
    assert res["exito"] is True
    assert env.exito is False
    assert res["pasos"] == 0
    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "E2_sam.json"
    e2.main(["--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=_mascaras_todo_en_cubo)
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert datos["exito_media"] == 0.0


def test_e_e2_sam_rapido_valida(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    e2_path = _repo_root() / "results" / "E2.json"
    out = tmp_path / "E2_sam.json"
    e2.main(["--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=_mascaras_honestas)
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    assert datos["config"]["estado"] == "sam"
    assert set(datos["sam"].keys()) == {"modo", "modelo", "prompt", "peticiones", "imagenes", "llamadas_red", "coste_estimado_usd", "falsos_exitos", "exitos_no_detectados", "latencia_ms_media"}
    assert "comparacion_e2" in datos
    for k in ("coincidencia_bot", "coincidencia_objetos", "coincidencia_todo", "pasos_ciegos"):
        assert k in datos["metricas"]
    assert datos["metricas"]["coincidencia_bot"]["media"] == 1.0
    ref = json.loads(e2_path.read_text(encoding="utf-8"))
    comp = datos["comparacion_e2"]
    assert comp["llamadas_llm_media"] == ref["metricas"]["llamadas_llm"]["por_seed"][0]
    assert comp["exito_media"] == ref["metricas"]["exito"]["por_seed"][0]


def test_f_sim_igual_sin_claves(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    o1 = tmp_path / "a.json"
    o2 = tmp_path / "b.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(o1), "--e1", str(e1_path)])
    e2.main(["--rapido", "--modo", "simulada", "--estado", "sim", "--out", str(o2), "--e1", str(e1_path)])
    d1 = json.loads(o1.read_text(encoding="utf-8"))
    d2 = json.loads(o2.read_text(encoding="utf-8"))
    d1.pop("fecha_utc")
    d2.pop("fecha_utc")
    assert d1 == d2
    for d in (d1, d2):
        assert "estado" not in d["config"]
        assert "sam" not in d
        assert "comparacion_e2" not in d


def test_g_max_red_no_toca_red(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "CLAVE-TEST-123")
    real = _texto_real()
    fake = _FakeTransporte([(200, {}, _sse_cuerpo([real], {"images_processed": 1}))])
    cli = ClienteSam(modo="api", cache_dir=str(tmp_path), transporte=fake, dormir=lambda s: None, max_red=1)
    f1 = np.zeros((8, 8, 3), dtype=np.uint8)
    f2 = np.ones((8, 8, 3), dtype=np.uint8)
    assert cli.segmentar([f1]) == real
    assert len(fake.llamadas) == 1
    assert cli.segmentar([f1]) == real
    assert len(fake.llamadas) == 1
    assert cli.llamadas_red == 1
    with pytest.raises(MetaAPIError):
        cli.segmentar([f2])
    assert len(fake.llamadas) == 1


def test_h_sam_simulada_sin_mascaras(tmp_path):
    with pytest.raises(ValueError):
        e2.main(["--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(tmp_path / "x.json")])


def test_i_replay_cache_vacio_cachemiss(tmp_path):
    vacio = tmp_path / "sam_vacio"
    vacio.mkdir()
    with pytest.raises(CacheMiss):
        e2.main(["--rapido", "--modo", "replay", "--estado", "sam", "--cache-sam", str(vacio), "--out", str(tmp_path / "o.json")])


def test_j_api_replay_sam_sin_red(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "CLAVE-TEST-123")
    e1_path = _repo_root() / "results" / "E1.json"
    cache_llm = tmp_path / "llm"
    cache_sam = tmp_path / "sam"
    out_api = tmp_path / "api.json"
    out_re = tmp_path / "re.json"
    doble = _TransporteDoble()
    e2.main(["--rapido", "--modo", "api", "--estado", "sam", "--cache", str(cache_llm), "--cache-sam", str(cache_sam), "--out", str(out_api), "--e1", str(e1_path)], transporte=doble)
    d_api = json.loads(out_api.read_text(encoding="utf-8"))
    validar(d_api, _schema())
    assert d_api["sam"]["imagenes"] > 0
    assert d_api["sam"]["llamadas_red"] > 0
    assert abs(d_api["coste_estimado_usd"] - (d_api["llm"]["coste_estimado_usd"] + d_api["sam"]["coste_estimado_usd"])) < 1e-12
    e2.main(["--rapido", "--modo", "replay", "--estado", "sam", "--cache", str(cache_llm), "--cache-sam", str(cache_sam), "--out", str(out_re), "--e1", str(e1_path)], transporte=_TransporteProhibido())
    d_re = json.loads(out_re.read_text(encoding="utf-8"))
    assert d_re["sam"]["llamadas_red"] == 0
    a = dict(d_api)
    b = dict(d_re)
    a.pop("fecha_utc")
    b.pop("fecha_utc")
    a["llm"] = dict(a["llm"])
    b["llm"] = dict(b["llm"])
    a["sam"] = dict(a["sam"])
    b["sam"] = dict(b["sam"])
    a["llm"].pop("modo")
    b["llm"].pop("modo")
    a["sam"].pop("modo")
    b["sam"].pop("modo")
    a["sam"].pop("llamadas_red")
    b["sam"].pop("llamadas_red")
    assert a == b
