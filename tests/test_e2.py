"""Anclas O3-E2: experimento con Atisbo en el lazo (PLIEGO §9). Sin red."""
from __future__ import annotations

import json
from pathlib import Path

from atisbo.exp import e2
from atisbo.exp.comun import generar_trazas, validar
from atisbo.estado import vector
from atisbo.lazo import Mundo


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema() -> dict:
    return json.loads((_repo_root() / "atisbo" / "exp" / "schema_results.json").read_text(encoding="utf-8"))


def test_g_rapido_simulada_valida(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "E2.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e1", str(e1_path)])
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    assert datos["experimento"] == "E2"
    assert isinstance(datos["h1"], bool)
    assert datos["llm"]["modo"] == "simulada"
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    comp = datos["comparacion_e1"]
    assert comp["llamadas_llm_media"] == e1["metricas"]["llamadas_llm"]["media"]
    assert comp["exito_media"] == e1["exito_media"]
    assert comp["latencia_ms_media"] == e1["metricas"]["latencia_ms"]["media"]
    assert comp["modo"] == e1["llm"]["modo"]
    assert comp["commit"] == e1["commit"]


def test_h_dos_rapidas_iguales_salvo_fecha(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out1), "--e1", str(e1_path)])
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out2), "--e1", str(e1_path)])
    d1 = json.loads(out1.read_text(encoding="utf-8"))
    d2 = json.loads(out2.read_text(encoding="utf-8"))
    d1.pop("fecha_utc")
    d2.pop("fecha_utc")
    assert d1 == d2


def test_estados_lazo_nuevos_no_son_de_entreno():
    s = 0
    assert e2.semilla_mundo(s) == 1000 + s
    entreno = generar_trazas(s, 200, 0.0)
    inicios = {tuple(float(x) for x in vector(ep[0][0]).ravel()) for ep in entreno}
    mundo = Mundo(seed=e2.semilla_mundo(s), p_perturb=0.0)
    for _ in range(10):
        inicio = mundo.reiniciar()
        v = tuple(float(x) for x in vector(inicio).ravel())
        assert v not in inicios
