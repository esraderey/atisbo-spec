"""Anclas O2-remates (PLIEGO S1/S5/S6/S8). Sin red: transporte falso o replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atisbo.control import PredictorRuido
from atisbo.estado import Accion, EstadoSim, vector
from atisbo.meta_api import ClienteMeta, MetaAPIError
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada, SYSTEM, construir_payload


def _estado(bot=(6, 6), paso=0):
    return EstadoSim(bot=bot, carga=None, objetos=((1, "rojo", 6, 6, False),),
                     cubos={"rojo": (0, 0), "verde": (11, 0), "azul": (0, 11)},
                     bloqueo=None, paso=paso)


class TransporteFalso:
    def __init__(self, cuerpos):
        self.cuerpos = list(cuerpos)
        self.llamadas = []
        self.cabeceras = []
        self.cuerpos_enviados = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append(url)
        self.cabeceras.append(dict(cabeceras))
        self.cuerpos_enviados.append(bytes(cuerpo))
        idx = min(len(self.llamadas) - 1, len(self.cuerpos) - 1)
        return self.cuerpos[idx]


def _ok(texto):
    body = json.dumps({"choices": [{"message": {"content": texto}}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
    return (200, {}, body)


def test_e1_trunca_api_y_replay_mismo_json(tmp_path, monkeypatch, capsys):
    from atisbo.exp import e1
    monkeypatch.setenv("META_API_KEY", "X")
    cache = tmp_path / "cache"
    fake = TransporteFalso([_ok('{"accion": "IZQ"}')] * 500)
    out_api = tmp_path / "E1api.json"
    r_api = e1.main(["--seed", "0", "--n", "2", "--p", "0.0", "--max-llm-calls", "165",
                     "--cache", str(cache), "--out", str(out_api), "--modo", "api"],
                    transporte=fake)
    d_api = json.loads(Path(r_api).read_text(encoding="utf-8"))
    assert d_api["truncada"] is True
    assert d_api["config"]["episodios_completados"] == 1
    assert d_api["config"]["n_episodios"] == 2
    assert d_api["metricas"]["llamadas_llm"]["media"] == pytest.approx(160.0)
    assert d_api["metricas"]["llamadas_llm"]["n"] == 1
    assert d_api["metricas"]["exito"]["media"] == pytest.approx(0.0)
    assert d_api["pasos_parcial"] == 5
    assert d_api["llm"]["llamadas"] == 165
    assert d_api["llm"]["peticiones"] == 165
    crudo = Path(r_api).read_text(encoding="utf-8")
    assert "llamadas_red" not in crudo
    salida_api = capsys.readouterr().out
    assert "red" in salida_api.lower()
    out_re = tmp_path / "E1re.json"
    r_re = e1.main(["--seed", "0", "--n", "2", "--p", "0.0", "--max-llm-calls", "165",
                    "--cache", str(cache), "--out", str(out_re), "--modo", "replay"],
                   transporte=None)
    d_re = json.loads(Path(r_re).read_text(encoding="utf-8"))
    assert d_re["truncada"] is True
    assert d_re["config"]["episodios_completados"] == 1
    assert d_re["pasos_parcial"] == 5
    a2 = dict(d_api)
    r2 = dict(d_re)
    a2.pop("fecha_utc")
    r2.pop("fecha_utc")
    a2["llm"] = {k: v for k, v in a2["llm"].items() if k != "modo"}
    r2["llm"] = {k: v for k, v in r2["llm"].items() if k != "modo"}
    assert a2 == r2


def test_simulada_igual_que_ruido_01_y_no_03():
    ests = [_estado((6, 6), i) for i in range(5)]
    for seed in (0, 1, 7):
        pol = PoliticaSimulada(seed=seed)
        ref = PredictorRuido(0.1, seed=seed)
        for es in ests:
            a_pol, _ = pol.decidir(es)
            ventana = np.asarray([vector(es)], dtype=np.float32)
            a_ref = ref.predecir(ventana).acciones[0]
            assert a_pol is a_ref
    pol01 = PoliticaSimulada(seed=123)
    ref03 = PredictorRuido(0.3, seed=123)
    dif = 0
    for i in range(50):
        es = _estado((i % 12, (i * 2) % 12), i)
        a_pol, _ = pol01.decidir(es)
        a_ref = ref03.predecir(np.asarray([vector(es)], dtype=np.float32)).acciones[0]
        if a_pol is not a_ref:
            dif += 1
    assert dif > 0


def test_rama_que_falla_exactamente_2_llamadas(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "X")
    fake = TransporteFalso([_ok("texto suelto"), _ok('{"accion": "VOLAR"}')])
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    pol = PoliticaLLM(cliente=cli)
    acc, _ = pol.decidir(_estado())
    assert acc is Accion.ESPERAR
    assert pol.llm_invalidas == 1
    assert len(fake.llamadas) == 2


def test_firma_y_reintentos(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "K")
    fake = TransporteFalso([_ok('{"accion": "IZQ"}')])
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    pol = PoliticaLLM(cliente=cli)
    pol.decidir(_estado())
    assert fake.llamadas[0] == "https://api.meta.ai/v1/chat/completions"
    cab = fake.cabeceras[0]
    assert cab["Authorization"] == "Bearer K"
    assert cab["Content-Type"] == "application/json"
    cuerpo = json.loads(fake.cuerpos_enviados[0].decode("utf-8"))
    assert cuerpo["model"] == "muse-spark-1.3"
    assert isinstance(cuerpo["messages"], list) and len(cuerpo["messages"]) >= 2
    assert cuerpo["response_format"]["type"] == "json_schema"
    assert cuerpo["response_format"]["json_schema"]["strict"] is True
    assert cuerpo["reasoning_effort"] == "minimal"
    esperas = []
    b429 = json.dumps({"error": {"message": "limite"}}).encode()
    fake2 = TransporteFalso([(429, {}, b429)] * 3)
    cli2 = ClienteMeta(modo="api", cache_dir=tmp_path / "c2", transporte=fake2,
                       dormir=lambda s: esperas.append(s))
    with pytest.raises(MetaAPIError):
        cli2.completar(construir_payload(_estado()))
    assert len(fake2.llamadas) == 3
    assert len(esperas) == 2
    esperas2 = []
    fake3 = TransporteFalso([(429, {"Retry-After": "7"}, b429), _ok('{"accion": "IZQ"}')])
    cli3 = ClienteMeta(modo="api", cache_dir=tmp_path / "c3", transporte=fake3,
                       dormir=lambda s: esperas2.append(s))
    cli3.completar(construir_payload(_estado()))
    assert esperas2 == [7.0]
    for i, st in enumerate((400, 401, 504)):
        fb = json.dumps({"error": {"message": "m", "param": "otro"}}).encode()
        fx = TransporteFalso([(st, {}, fb)])
        cx = ClienteMeta(modo="api", cache_dir=tmp_path / f"cs{st}", transporte=fx,
                         dormir=lambda s: None)
        with pytest.raises(MetaAPIError):
            cx.completar(construir_payload(_estado()))
        assert len(fx.llamadas) == 1, st


def test_sin_degradacion_y_huella_es_lo_enviado(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "K")
    fb = json.dumps({"error": {"message": "bad effort", "param": "reasoning_effort"}}).encode()
    fx = TransporteFalso([(400, {}, fb)])
    cx = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fx, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cx.completar(construir_payload(_estado()))
    assert len(fx.llamadas) == 1
    enviado = json.loads(fx.cuerpos_enviados[0].decode("utf-8"))
    assert enviado["reasoning_effort"] == "minimal"


def test_clave_con_espacios_y_control(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "  CLAVE-BUENA-1  \r\n")
    fake = TransporteFalso([_ok('{"accion": "IZQ"}')])
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    cli.completar(construir_payload(_estado()))
    assert fake.cabeceras[0]["Authorization"] == "Bearer CLAVE-BUENA-1"
    monkeypatch.setenv("META_API_KEY", "MA\x01LA")
    fake2 = TransporteFalso([_ok('{"accion": "IZQ"}')])
    cli2 = ClienteMeta(modo="api", cache_dir=tmp_path / "c2", transporte=fake2, dormir=lambda s: None)
    with pytest.raises(MetaAPIError) as ei:
        cli2.completar(construir_payload(_estado()))
    assert "MA\x01LA" not in str(ei.value)
    assert len(fake2.llamadas) == 0


def test_200_invalido_reintenta_no_aborta(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "X")
    sin = json.dumps({"usage": {}}).encode()
    nul = json.dumps({"choices": [{"message": {"content": None}}], "usage": {}}).encode()
    for i, bruto in enumerate((sin, nul)):
        fx = TransporteFalso([(200, {}, bruto), _ok('{"accion": "DER"}')])
        cx = ClienteMeta(modo="api", cache_dir=tmp_path / f"ci{i}", transporte=fx,
                         dormir=lambda s: None)
        pol = PoliticaLLM(cliente=cx)
        acc, _ = pol.decidir(_estado())
        assert acc is Accion.DER
        assert len(fx.llamadas) == 2
    fx2 = TransporteFalso([(200, {}, sin), (200, {}, nul)])
    cx2 = ClienteMeta(modo="api", cache_dir=tmp_path / "cc", transporte=fx2, dormir=lambda s: None)
    pol2 = PoliticaLLM(cliente=cx2)
    acc2, _ = pol2.decidir(_estado())
    assert acc2 is Accion.ESPERAR
    assert pol2.llm_invalidas == 1
    assert len(fx2.llamadas) == 2


def test_tope_cuenta_cache_y_no_suma_invalida_en_reintento(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "X")
    fake = TransporteFalso([_ok('{"accion": "IZQ"}')] * 10)
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    pol = PoliticaLLM(cliente=cli, max_llm_calls=1)
    ea = _estado((6, 6), 0)
    acc, _ = pol.decidir(ea)
    assert acc is Accion.IZQ
    assert len(fake.llamadas) == 1
    eb = _estado((6, 6), 0)
    acc2, _ = pol.decidir(eb)
    assert acc2 is Accion.ESPERAR
    assert pol.truncada is True
    cli_r = ClienteMeta(modo="replay", cache_dir=tmp_path)
    pol_r = PoliticaLLM(cliente=cli_r, max_llm_calls=1)
    acc_r, _ = pol_r.decidir(ea)
    assert acc_r is Accion.IZQ
    acc_r2, _ = pol_r.decidir(eb)
    assert acc_r2 is Accion.ESPERAR
    assert pol_r.truncada is True
    fake3 = TransporteFalso([_ok("mal"), _ok('{"accion": "DER"}')])
    cli3 = ClienteMeta(modo="api", cache_dir=tmp_path / "c3", transporte=fake3, dormir=lambda s: None)
    pol3 = PoliticaLLM(cliente=cli3, max_llm_calls=1)
    acc3, _ = pol3.decidir(_estado())
    assert acc3 is Accion.ESPERAR
    assert pol3.truncada is True
    assert pol3.llm_invalidas == 0
    assert len(fake3.llamadas) == 1


def test_correr_episodio_solo_episodio_y_metricas_s5(tmp_path, monkeypatch):
    from atisbo.control import PredictorNulo
    from atisbo.lazo import Mundo, correr_episodio
    monkeypatch.setenv("META_API_KEY", "X")
    seq = [_ok("texto suelto"), _ok('{"accion": "VOLAR"}')] + [_ok('{"accion": "IZQ"}')] * 600
    fake = TransporteFalso(seq)
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    pol = PoliticaLLM(cliente=cli, max_llm_calls=10000)
    mundo = Mundo(seed=0, p_perturb=0.0)
    r1 = correr_episodio(PredictorNulo(), pol, mundo, tau=0.5, epsilon=1e9, registro=[])
    assert r1["llm_invalidas"] == 1
    assert r1["racha_media"] == 0
    assert r1["malas_aceptadas"] == 0
    assert r1["tasa_malas_aceptadas"] == 0
    assert r1["llamadas_llm"] == r1["pasos"]
    r2 = correr_episodio(PredictorNulo(), pol, mundo, tau=0.5, epsilon=1e9, registro=[])
    assert r2["llm_invalidas"] == 0
    assert r2["racha_media"] == 0
    assert r2["malas_aceptadas"] == 0
    assert r2["tasa_malas_aceptadas"] == 0
    assert int(getattr(pol, "llm_invalidas", 0)) == 1


def test_prompt_trae_faltante_s1():
    t = SYSTEM.lower()
    assert "en_cubo" in SYSTEM and "false" in t
    assert "id" in t and "nuevo" in t
    assert "perturba" in t
    assert "5 pasos" in t or "cinco" in t or "durante 5" in t


def test_cache_atomica_sin_medias(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "X")
    fake = TransporteFalso([_ok('{"accion": "IZQ"}')])
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    cli.completar(construir_payload(_estado()))
    f = list(Path(tmp_path).glob("*.json"))
    assert len(f) == 1
    json.loads(f[0].read_text(encoding="utf-8"))
    assert list(Path(tmp_path).glob("*.tmp")) == []
