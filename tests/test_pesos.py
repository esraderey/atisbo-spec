"""El GRU publicado: pesos fijos reproducibles, datos incluidos y apagado por defecto en la consola."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from atisbo.control import PredictorNulo
from atisbo.exp.comun import construir_muestras, generar_trazas, split_episodios
from atisbo.lazo import K
from atisbo.modelo import Atisbo
from demo.pesos import entrenar_publicado
from demo.vivo import Consola, _parse_args, cargar_predictor
from atisbo.lazo import Mundo
from atisbo.politica_llm import PoliticaSimulada

RAIZ = Path(__file__).resolve().parents[1] / "modelo"
MANIFIESTO = json.loads((RAIZ / "manifiesto.json").read_text(encoding="utf-8"))


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def test_manifiesto_coincide_con_los_archivos():
    assert _sha(RAIZ / "atisbo_seed0.pt") == MANIFIESTO["pesos_sha256"]
    assert _sha(RAIZ / "datos_entrenamiento.npz") == MANIFIESTO["datos_sha256"]


def test_datos_publicados_son_los_del_entrenamiento():
    tr, _ = split_episodios(generar_trazas(0, 200, 0.02), 0)
    X, A, S = construir_muestras(tr, K)
    d = np.load(RAIZ / "datos_entrenamiento.npz")
    assert np.array_equal(d["X"], X) and np.array_equal(d["A"], A) and np.array_equal(d["S"], S)


def test_reentrenar_da_los_pesos_publicados():
    nuevo, _ = entrenar_publicado()
    fijo = Atisbo.cargar(RAIZ / "atisbo_seed0.pt")
    a, b = nuevo.state_dict(), fijo.state_dict()
    assert set(a) == set(b) and all(a[k].equal(b[k]) for k in a)
    assert fijo.n_parametros() == MANIFIESTO["parametros"]


def test_consola_apagada_por_defecto():
    a = _parse_args([])
    assert a.atisbo is False
    predictor, _eps = cargar_predictor(False)
    assert isinstance(predictor, PredictorNulo)
    assert _parse_args(["--atisbo"]).atisbo is True


def test_encendida_usa_los_pesos_fijos():
    predictor, eps = cargar_predictor(True)
    assert isinstance(predictor, Atisbo) and eps == MANIFIESTO["epsilon"]


def test_apagado_todo_lo_decide_la_politica_y_la_pagina_lo_sabe():
    predictor, eps = cargar_predictor(False)
    c = Consola(predictor, PoliticaSimulada(seed=1), Mundo(seed=3, p_perturb=0.0), 0.5, eps, "simulada", pausa=0.0, con_atisbo=False)
    c.episodio()
    ev = c.bus.desde(0)
    assert ev[0]["atisbo"] is False
    pasos = [e for e in ev if e["tipo"] == "paso" and "fuente" in e]
    assert pasos and all(e["fuente"] == "llm" for e in pasos)


def test_la_pagina_dice_que_atisbo_no_es_de_meta():
    html = (Path(__file__).resolve().parents[1] / "demo" / "vivo.html").read_text(encoding="utf-8")
    assert "not a Meta model" in html and "Atisbo off" in html


def test_readme_documenta_el_gru():
    texto = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for t in ("not a Meta model", "modelo/atisbo_seed0.pt", "modelo/datos_entrenamiento.npz", "demo.pesos", "--atisbo"):
        assert t in texto
