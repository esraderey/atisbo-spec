"""Consola en vivo, sin red: eventos, perturbación a petición, protección de la clave y servidor."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from atisbo.control import PredictorOraculo, PredictorRuido
from atisbo.lazo import Mundo
from atisbo.meta_api import ClienteMeta
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada
from demo.vivo import Bus, Consola, MundoVivo, _ClienteVisto, capa_mascaras, crear_servidor


class TransporteFalso:
    def post(self, url, cabeceras, cuerpo):
        r = {"choices": [{"message": {"content": "{\"accion\": \"ESPERAR\"}"}}], "usage": {"prompt_tokens": 500, "completion_tokens": 20}}
        return 200, {}, json.dumps(r).encode("utf-8")


def _consola(predictor=None, politica=None, seed=3, **kw):
    return Consola(predictor or PredictorRuido(0.3, seed=1), politica or PoliticaSimulada(seed=1), Mundo(seed=seed, p_perturb=0.0), 0.5, 1e9, "simulada", pausa=0.0, **kw)


def _mascaras_falsas(frame):
    m = np.zeros((384, 384), dtype=bool)
    m[40:57, 40:57] = True
    return {1: m}


def test_bus_numera_y_filtra():
    b = Bus()
    b.publica(tipo="a")
    b.publica(tipo="b")
    assert [e["n"] for e in b.desde(0)] == [0, 1]
    assert [e["tipo"] for e in b.desde(1)] == ["b"]


def test_episodio_publica_un_paso_por_accion():
    c = _consola()
    c.episodio()
    ev = c.bus.desde(0)
    pasos = [e for e in ev if e["tipo"] == "paso"]
    fin = next(e for e in ev if e["tipo"] == "fin")
    assert [e["tipo"] for e in ev[:2]] == ["listo", "inicio"] and len(pasos) == fin["pasos"] + 1
    assert "fuente" not in pasos[0] and all(p["fuente"] in ("llm", "atisbo") for p in pasos[1:])
    assert sum(1 for p in pasos[1:] if p["fuente"] == "llm") == fin["llamadas"]


def test_perturbar_solo_cuando_se_pide():
    bus = Bus()
    m1, m2 = Mundo(seed=4, p_perturb=0.0), Mundo(seed=4, p_perturb=0.0)
    v = MundoVivo(m1, bus)
    s1, s2 = v.reiniciar(), m2.reiniciar()
    from atisbo.estado import Accion

    a1 = v.avanzar(s1, Accion.ESPERAR)
    a2 = m2.avanzar(s2, Accion.ESPERAR)
    assert a1 == a2 and bus.desde(0) == []
    v.perturbar()
    b1 = v.avanzar(a1, Accion.ESPERAR)
    b2 = m2.avanzar(a2, Accion.ESPERAR)
    assert b1.objetos != b2.objetos and b1.bot == b2.bot and len(b1.objetos) == len(b2.objetos)
    assert [e["tipo"] for e in bus.desde(0)] == ["perturbacion"]
    assert v.pendiente is False
    c1 = v.avanzar(b1, Accion.ESPERAR)
    assert c1.objetos == b1.objetos and len(bus.desde(0)) == 1


def test_con_sam_publica_imagen_y_mascaras():
    c = _consola(mascaras_de=_mascaras_falsas)
    c.episodio()
    ev = c.bus.desde(0)
    sam = [e for e in ev if e["tipo"] == "llamada" and e.get("mascaras") == 1]
    pasos = [e for e in ev if e["tipo"] == "paso"]
    assert len(sam) == len(pasos) and all(e["modelo"] == "local stand-in" and "prompt" not in e for e in sam)
    assert all(p["mascaras"] and p["frame"] for p in pasos)


def test_capa_tine_por_tamano():
    frame = np.full((384, 384, 3), 255, dtype=np.uint8)
    salida = capa_mascaras(frame, _mascaras_falsas(frame))
    assert salida[45, 45].tolist() != [255, 255, 255] and salida[100, 100].tolist() == [255, 255, 255]
    assert salida[45, 45][2] > salida[45, 45][0]


def test_llamada_real_visible_y_sin_la_clave(tmp_path, monkeypatch):
    clave = "clave-de-prueba-que-no-debe-salir"
    monkeypatch.setenv("META_API_KEY", clave)
    cliente = ClienteMeta(modo="api", cache_dir=str(tmp_path), transporte=TransporteFalso())
    c = _consola(predictor=PredictorRuido(0.9, seed=1))
    c.politica = PoliticaLLM(cliente=_ClienteVisto(cliente, c.bus), max_llm_calls=5)
    c.episodio()
    ev = c.bus.desde(0)
    ll = [e for e in ev if e["tipo"] == "llamada"]
    assert ll and ll[0]["modelo"] == "muse-spark-1.3" and ll[0]["red"] is True
    assert ll[0]["entrada"] == 500 and ll[0]["salida"] == 20 and "ESPERAR" in ll[0]["respuesta"]
    assert len(ll) <= 5
    assert clave not in json.dumps(ev)


def test_error_no_filtra_la_clave(monkeypatch):
    from atisbo.meta_api import MetaAPIError

    clave = "otra-clave-de-prueba"
    monkeypatch.setenv("META_API_KEY", clave)

    class Rota:
        truncada = False

        def decidir(self, s):
            raise MetaAPIError("fallo con " + clave)

    c = _consola(predictor=PredictorRuido(0.9, seed=1), politica=Rota())
    c.episodio()
    err = [e for e in c.bus.desde(0) if e["tipo"] == "error"]
    assert err and clave not in err[0]["texto"] and c.corriendo is False


def test_oraculo_no_llama_a_muse_spark():
    c = _consola(predictor=PredictorOraculo())
    c.episodio()
    fin = next(e for e in c.bus.desde(0) if e["tipo"] == "fin")
    assert fin["exito"] is True and fin["llamadas"] == 0


@pytest.fixture
def servidor():
    c = _consola()
    srv = crear_servidor(c, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield c, srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _pide(puerto, ruta, metodo="GET", cabeceras=None):
    req = urllib.request.Request(f"http://127.0.0.1:{puerto}{ruta}", method=metodo, headers=cabeceras or {}, data=b"" if metodo == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_servidor_solo_en_local(servidor):
    _c, puerto = servidor
    codigo, cuerpo = _pide(puerto, "/")
    assert codigo == 200 and b"ATISBO-SPEC" in cuerpo
    assert [e["tipo"] for e in json.loads(_pide(puerto, "/eventos?desde=0")[1])] == ["listo"]
    assert _pide(puerto, "/", cabeceras={"Host": "evil.example"})[0] == 403
    assert _pide(puerto, "/nada")[0] == 404


def test_post_exige_cabecera(servidor):
    c, puerto = servidor
    assert _pide(puerto, "/perturbar", "POST")[0] == 403
    assert c.vivo.pendiente is False
    assert _pide(puerto, "/perturbar", "POST", {"X-Consola": "1"})[0] == 200
    assert c.vivo.pendiente is True
