"""Página en vivo: los datos salen del replay de E2 y la página no depende de nada externo."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atisbo.env.trazas import CUBOS_FIJOS
from atisbo.estado import EstadoSim
from demo.pagina import perturbaciones
from demo.registro import estado_a_dict

DATOS = Path("results/pagina.json")
PAGINA = Path("index.html")


def _cargar(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _estado(objetos, bot=(5, 5), paso=0, bloqueo=None):
    return EstadoSim(bot=bot, carga=None, objetos=objetos, cubos=dict(CUBOS_FIJOS), bloqueo=bloqueo, paso=paso)


def test_episodios_igual_que_e2():
    d = _cargar(DATOS)
    e2 = _cargar("results/E2.json")["metricas"]
    eps = d["episodios"]
    assert d["seed"] == 0 and len(eps) == 10
    assert sum(e["llamadas_llm"] for e in eps) / 10 == pytest.approx(e2["llamadas_llm"]["por_seed"][0])
    assert sum(1.0 if e["exito"] else 0.0 for e in eps) / 10 == pytest.approx(e2["exito"]["por_seed"][0])
    assert sum(len(e["pasos"]) for e in eps) / 10 == pytest.approx(e2["pasos"]["por_seed"][0])


def test_episodio_4_es_el_del_video():
    ep = _cargar(DATOS)["episodios"][4]
    reg = _cargar("results/registro_E2_seed0.json")
    assert [p["estado"] for p in ep["pasos"]] == [p["estado"] for p in reg["pasos"]]
    assert [p["accion"] for p in ep["pasos"]] == [p["accion"] for p in reg["pasos"]]
    assert ep["llamadas_llm"] == reg["llamadas_llm"] and ep["exito"] == reg["exito"]
    assert ep["perturbaciones"] == [58]


def test_incluye_el_episodio_que_falla():
    eps = _cargar(DATOS)["episodios"]
    assert eps[0]["exito"] is False and len(eps[0]["pasos"]) == 160


def test_perturbaciones_detecta_salto_y_bloqueo():
    a = _estado(((1, "rojo", 3, 3, False),))
    salto = _estado(((1, "rojo", 4, 3, False),), paso=1)
    quieto = _estado(((1, "rojo", 3, 3, False),), paso=1)
    bloqueado = _estado(((1, "rojo", 3, 3, False),), paso=1, bloqueo=(7, 7))
    paso = {"estado": estado_a_dict(a), "accion": "ESPERAR"}
    assert perturbaciones([paso], estado_a_dict(salto)) == [1]
    assert perturbaciones([paso], estado_a_dict(quieto)) == []
    assert perturbaciones([paso], estado_a_dict(bloqueado)) == [1]


def test_quitar_un_bloqueo_no_es_perturbacion():
    con = _estado(((1, "rojo", 3, 3, False),), bloqueo=(7, 7))
    sin = _estado(((1, "rojo", 3, 3, False),), paso=1)
    assert perturbaciones([{"estado": estado_a_dict(con), "accion": "ESPERAR"}], estado_a_dict(sin)) == []


def test_pagina_sin_dependencias_externas_ni_claves():
    t = PAGINA.read_text(encoding="utf-8")
    assert "results/pagina.json" in t and "results/resumen.json" in t
    assert re.search(r"<(script|link|img)[^>]+(src|href)=[\"']https?:", t) is None
    assert re.search(r"(?i)api[_-]?key|bearer |authorization", t) is None
