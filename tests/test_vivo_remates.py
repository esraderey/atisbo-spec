"""Anclas de la consola en vivo: lo que colgaría la página o la haría mentir ante quien mira."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from atisbo.control import PredictorOraculo, PredictorRuido
from atisbo.env.perturbaciones import celdas_vacias
from atisbo.estado import Accion
from atisbo.lazo import Mundo
from atisbo.meta_api import ClienteMeta
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada
from atisbo.sam import ClienteSam, TopeSam
from demo.vivo import Bus, Consola, MundoVivo, _ClienteVisto, mascaras_por_color, mover_el_cercano, pedir_clave


def _texto_sam() -> str:
    raiz = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sam"
    for f in sorted(raiz.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("payload", {}).get("medio") == "imagen":
            return str(d["texto"])
    raise AssertionError("sin fixture imagen")


def _sse(texto: str) -> bytes:
    lineas = ["data: " + json.dumps({"type": "response.output_text.delta", "delta": texto})]
    lineas.append("data: " + json.dumps({"type": "response.completed", "response": {"usage": {"images_processed": 1}}}))
    return ("\n".join(lineas) + "\n").encode("utf-8")


class TransporteMeta:
    """Responde como api.meta.ai a los dos modelos, sin red."""

    def __init__(self) -> None:
        self.urls: list = []
        self.cabeceras: list = []

    def post(self, url, cabeceras, cuerpo):
        self.urls.append(url)
        self.cabeceras.append(cabeceras)
        if "chat/completions" in url:
            r = {"choices": [{"message": {"content": "{\"accion\":\"ESPERAR\"}"}}], "usage": {"prompt_tokens": 400, "completion_tokens": 9}}
            return 200, {}, json.dumps(r).encode("utf-8")
        return 200, {}, _sse(_texto_sam())


def _consola(predictor, politica=None, mundo=None, epsilon=1e9, **kw):
    return Consola(predictor, politica or PoliticaSimulada(seed=1), mundo or Mundo(seed=3, p_perturb=0.0), 0.5, epsilon, "simulada", pausa=0.0, **kw)


class PredictorEmpuja:
    """Oráculo que aprieta el botón de perturbar en su predicción n."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.vistas = 0
        self.consola = None
        self.oraculo = PredictorOraculo()

    def predecir(self, ventana):
        self.vistas += 1
        if self.vistas == self.n:
            self.consola.vivo.perturbar()
        return self.oraculo.predecir(ventana)


def test_la_sorpresa_llega_a_la_pagina():
    p = PredictorEmpuja(3)
    c = _consola(p, epsilon=0.0)
    p.consola = c
    c.episodio()
    ev = c.bus.desde(0)
    tipos = [e["tipo"] for e in ev]
    assert "perturbacion" in tipos
    tras = [e for e in ev[tipos.index("perturbacion"):] if e["tipo"] == "paso"]
    sorpresas = [e for e in tras if e.get("sorpresa")]
    assert sorpresas and sorpresas[0]["fuente"] == "llm"
    assert not any(e.get("sorpresa") for e in ev[:tipos.index("perturbacion")])


def test_confianza_nan_no_rompe_el_json():
    class PredictorNan:
        def predecir(self, ventana):
            return dataclasses.replace(PredictorOraculo().predecir(ventana), confianza=float("nan"))

    c = _consola(PredictorNan())
    c.episodio()
    texto = json.dumps(c.bus.desde(0), allow_nan=False)
    assert "\"fuente\": \"llm\"" in texto


def test_perturbacion_natural_se_anuncia():
    c = _consola(PredictorRuido(0.0, seed=1), mundo=Mundo(seed=3, p_perturb=1.0))
    c.episodio()
    naturales = [e for e in c.bus.desde(0) if e["tipo"] == "perturbacion"]
    assert naturales and all(e["pedida"] is False and e["ok"] is True for e in naturales)


def test_boton_senala_la_casilla_movida():
    bus = Bus()
    v = MundoVivo(Mundo(seed=4, p_perturb=0.0), bus)
    s = v.reiniciar()
    v.perturbar()
    s2 = v.avanzar(s, Accion.ESPERAR)
    movidos = [o for o in s2.objetos if o not in s.objetos]
    assert len(movidos) == 1 and bus.desde(0)[0]["celda"] == [movidos[0][2], movidos[0][3]]
    dist = lambda x, y: abs(x - s2.bot[0]) + abs(y - s2.bot[1])
    antes = next(o for o in s.objetos if o[0] == movidos[0][0])
    assert dist(antes[2], antes[3]) == min(dist(o[2], o[3]) for o in s.objetos if not o[4])
    assert dist(movidos[0][2], movidos[0][3]) == max(dist(*c) for c in celdas_vacias(s2) + [(movidos[0][2], movidos[0][3])])


def test_sin_objetos_libres_el_boton_no_inventa():
    s = dataclasses.replace(Mundo(seed=4, p_perturb=0.0).reiniciar(), objetos=())
    assert mover_el_cercano(s) == (s, False)


def test_boton_pulsado_no_cruza_de_episodio():
    c = _consola(PredictorOraculo())
    c.vivo.perturbar()
    c.episodio()
    assert [e for e in c.bus.desde(0) if e["tipo"] == "perturbacion"] == []


def test_error_inesperado_llega_a_la_pagina_sin_la_clave(monkeypatch):
    clave = "clave-de-prueba-inesperada"
    monkeypatch.setenv("META_API_KEY", "  " + clave + "\n")

    class Rota:
        def decidir(self, s):
            raise ValueError("fallo raro con " + clave)

    c = _consola(PredictorRuido(0.9, seed=1), politica=Rota())
    c.corriendo = True
    c.episodio()
    err = [e for e in c.bus.desde(0) if e["tipo"] == "error"]
    assert err and "fallo raro" in err[0]["texto"] and clave not in err[0]["texto"]
    assert c.corriendo is False


def test_mascaras_por_color_ve_lo_mismo_que_el_mundo():
    c = _consola(PredictorRuido(0.1, seed=1), mascaras_de=mascaras_por_color)
    c.episodio()
    fin = next(e for e in c.bus.desde(0) if e["tipo"] == "fin")
    assert c.mundo.frames > 1 and fin["exito"] is True
    assert c.mundo.coincidencias()["bot"] == 1.0


def _consola_api(tmp_path, monkeypatch, max_red=None):
    clave = "clave-de-prueba-del-camino-api"
    monkeypatch.setenv("META_API_KEY", clave)
    t = TransporteMeta()
    bus = Bus()
    meta = ClienteMeta(modo="api", cache_dir=str(tmp_path / "llm"), transporte=t)
    sam = ClienteSam(modo="api", cache_dir=str(tmp_path / "sam"), transporte=t, max_red=max_red)
    politica = PoliticaLLM(cliente=_ClienteVisto(meta, bus), max_llm_calls=4)
    c = Consola(PredictorRuido(0.9, seed=1), politica, Mundo(seed=3, p_perturb=0.0), 0.5, 1e9, "api", bus=bus, cliente_sam=sam, pausa=0.0)
    return c, t, clave


def test_camino_api_completo_con_los_dos_modelos(tmp_path, monkeypatch):
    c, t, clave = _consola_api(tmp_path, monkeypatch)
    c.episodio()
    ev = c.bus.desde(0)
    llamadas = [e for e in ev if e["tipo"] == "llamada"]
    assert ev[0]["tipo"] == "listo" and ev[0]["modo"] == "api" and ev[0]["sam"] is True
    assert {e["modelo"] for e in llamadas} == {"muse-spark-1.3", "sam-3.1"}
    assert llamadas[0]["modelo"] == "sam-3.1" and llamadas[0]["red"] is True and llamadas[0]["prompt"] == "colored square"
    primera_llm = next(e for e in llamadas if e["modelo"] == "muse-spark-1.3")
    assert primera_llm["red"] is True and primera_llm["entrada"] == 400 and "ESPERAR" in primera_llm["respuesta"]
    assert any("chat/completions" in u for u in t.urls) and any("/responses" in u for u in t.urls)
    assert all(c2["Authorization"] == "Bearer " + clave for c2 in t.cabeceras)
    assert clave not in json.dumps(ev)
    fin = next(e for e in ev if e["tipo"] == "fin")
    assert fin["truncada"] is True


def test_tope_sam_se_avisa_y_libera_la_consola(tmp_path, monkeypatch):
    c, _t, clave = _consola_api(tmp_path, monkeypatch, max_red=0)
    c.corriendo = True
    c.episodio()
    err = [e for e in c.bus.desde(0) if e["tipo"] == "error"]
    assert err and "tope" in err[0]["texto"].lower() and c.corriendo is False
    assert issubclass(TopeSam, Exception) and clave not in err[0]["texto"]


def test_la_clave_se_pide_sin_eco_solo_si_falta(monkeypatch):
    pedidas = []
    monkeypatch.setattr("demo.vivo.getpass.getpass", lambda prompt: pedidas.append(prompt) or "clave-tecleada")
    monkeypatch.setenv("META_API_KEY", "clave-del-entorno")
    pedir_clave()
    assert pedidas == [] and os.environ["META_API_KEY"] == "clave-del-entorno"
    monkeypatch.delenv("META_API_KEY")
    pedir_clave()
    assert pedidas == ["META_API_KEY: "] and os.environ["META_API_KEY"] == "clave-tecleada"


def test_sin_clave_no_arranca(monkeypatch):
    monkeypatch.setenv("META_API_KEY", " ")
    monkeypatch.setattr("demo.vivo.getpass.getpass", lambda prompt: "  ")
    with pytest.raises(SystemExit):
        pedir_clave()


@pytest.mark.parametrize("etiqueta", ["Muse Spark", "Simulated policy", "Surprise", "Low confidence", "lost the bot"])
def test_la_pagina_distingue_quien_decide_y_por_que(etiqueta):
    html = (Path(__file__).resolve().parents[1] / "demo" / "vivo.html").read_text(encoding="utf-8")
    assert etiqueta in html
