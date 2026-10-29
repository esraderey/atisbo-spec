"""Anclas O2 (PLIEGO §6): cliente Meta caché, parser política, simulada, tope. Sin red."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from atisbo.control import PredictorRuido
from atisbo.env.rutina import elegir_accion
from atisbo.estado import Accion, EstadoSim
from atisbo.meta_api import CacheMiss, ClienteMeta
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada, parsear_accion


def _payload():
    return {
        "model": "muse-spark-1.3",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": '{"bot": [6, 6]}'},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "accion",
                "schema": {"type": "object", "properties": {"accion": {"enum": ["ARRIBA"]}}, "required": ["accion"]},
                "strict": True,
            },
        },
        "reasoning_effort": "minimal",
    }


def _huella(payload):
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def _estado():
    return EstadoSim(bot=(6, 6), carga=None, objetos=((1, "rojo", 6, 6, False),),
                     cubos={"rojo": (0, 0), "verde": (11, 0), "azul": (0, 11)}, bloqueo=None, paso=0)


class TransporteFalso:
    def __init__(self, cuerpos):
        self.cuerpos = list(cuerpos)
        self.llamadas = []
        self.headers_vistos = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append((url, cuerpo))
        self.headers_vistos.append(dict(cabeceras))
        status, h, b = self.cuerpos[min(len(self.llamadas) - 1, len(self.cuerpos) - 1)]
        return status, h, b


def _resp_ok(texto):
    body = json.dumps({"choices": [{"message": {"content": texto}}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode("utf-8")
    return (200, {}, body)


def test_replay_sirve_por_huella_canonica_y_falta_es_cachemiss(tmp_path):
    payload = _payload()
    huella = _huella(payload)
    f = Path(tmp_path) / (huella + ".json")
    f.write_text(json.dumps({"payload": payload,
                             "cuerpo": {"choices": [{"message": {"content": '{"accion": "IZQ"}'}}], "usage": {}},
                             "latencia_ms": 12.5}), encoding="utf-8")
    cli = ClienteMeta(modo="replay", cache_dir=tmp_path)
    reordenado = {k: payload[k] for k in reversed(list(payload.keys()))}
    texto, _uso, lat = cli.completar(reordenado)
    assert json.loads(texto) == {"accion": "IZQ"}
    assert lat == 12.5
    with pytest.raises(CacheMiss):
        cli.completar({"model": "muse-spark-1.3", "otro": 1})


def test_api_graba_cache_sin_clave_y_reusa_sin_red(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "CLAVE-FALSA-TEST-123")
    fake = TransporteFalso([_resp_ok('{"accion": "IZQ"}')])
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    payload = _payload()
    cli.completar(payload)
    assert len(fake.llamadas) == 1
    archivos = list(Path(tmp_path).glob("*.json"))
    assert len(archivos) == 1
    crudo = archivos[0].read_text(encoding="utf-8")
    assert "CLAVE-FALSA-TEST-123" not in crudo
    assert "Bearer" not in crudo
    data = json.loads(crudo)
    assert set(data.keys()) == {"payload", "cuerpo", "latencia_ms"}
    cli.completar(payload)
    assert len(fake.llamadas) == 1


def test_parser_acepta_rechaza_reintenta_y_cae(tmp_path, monkeypatch):
    assert parsear_accion('{"accion": "IZQ"}') is Accion.IZQ
    with pytest.raises(ValueError):
        parsear_accion('{"accion": "VOLAR"}')
    with pytest.raises(ValueError):
        parsear_accion('voy a la izquierda')
    monkeypatch.setenv("META_API_KEY", "X")
    fake = TransporteFalso([_resp_ok('{"accion": "VOLAR"}'), _resp_ok('{"accion": "DER"}')])
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    pol = PoliticaLLM(cliente=cli)
    acc, _lat = pol.decidir(_estado())
    assert acc is Accion.DER
    assert len(fake.llamadas) == 2
    fake2 = TransporteFalso([_resp_ok('texto suelto'), _resp_ok('{"accion": "VOLAR"}')])
    cli2 = ClienteMeta(modo="api", cache_dir=tmp_path / "c2", transporte=fake2, dormir=lambda s: None)
    pol2 = PoliticaLLM(cliente=cli2)
    acc2, _lat2 = pol2.decidir(_estado())
    assert acc2 is Accion.ESPERAR
    assert pol2.llm_invalidas == 1


def test_simulada_es_ruido_01_determinista():
    p1 = PoliticaSimulada(seed=7)
    p2 = PoliticaSimulada(seed=7)
    assert p1.p == 0.1
    est = _estado()
    seq1 = [p1.decidir(est)[0] for _ in range(20)]
    seq2 = [p2.decidir(est)[0] for _ in range(20)]
    assert seq1 == seq2
    r1 = PredictorRuido(0.1, seed=7)
    assert isinstance(p1._pred, PredictorRuido)


def _estado_variado(bot, paso):
    return EstadoSim(bot=bot, carga=None, objetos=((1, "rojo", 6, 6, False),),
                     cubos={"rojo": (0, 0), "verde": (11, 0), "azul": (0, 11)},
                     bloqueo=None, paso=paso)


def test_tope_max_llm_calls_limita_red_y_reusa_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "X")
    fake = TransporteFalso([_resp_ok('{"accion": "IZQ"}')] * 10)
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    pol = PoliticaLLM(cliente=cli, max_llm_calls=2)
    ea = _estado_variado((6, 6), 0)
    eb = _estado_variado((5, 6), 1)
    ec = _estado_variado((4, 6), 2)
    acc_a, _ = pol.decidir(ea)
    assert acc_a is Accion.IZQ
    acc_b, _ = pol.decidir(eb)
    assert acc_b is Accion.IZQ
    assert len(fake.llamadas) == 2
    assert pol.llamadas_llm == 2
    assert pol.llamadas_red == 2
    assert pol.peticiones == 2
    assert pol.truncada is False
    acc_re, _ = pol.decidir(ea)
    assert acc_re is Accion.ESPERAR
    assert len(fake.llamadas) == 2
    assert pol.llamadas_llm == 3
    assert pol.llamadas_red == 2
    assert pol.peticiones == 2
    assert pol.truncada is True
    acc_c, _ = pol.decidir(ec)
    assert acc_c is Accion.ESPERAR
    assert pol.truncada is True
    assert len(fake.llamadas) == 2
    assert pol.llamadas_llm == 4
    assert pol.llamadas_red == 2
