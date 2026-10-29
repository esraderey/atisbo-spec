"""Anclas O4-E3: calibración conforme + lazo por alfa (PLIEGO §7 y §9). Sin red."""
from __future__ import annotations

import json
from pathlib import Path

from atisbo.exp import e3
from atisbo.exp.comun import validar


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema() -> dict:
    return json.loads((_repo_root() / "atisbo" / "exp" / "schema_results.json").read_text(encoding="utf-8"))


def _e1_repo() -> Path:
    return _repo_root() / "results" / "E1.json"


def test_f_rapido_simulada_valida(tmp_path):
    out = tmp_path / "E3.json"
    e3.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e1", str(_e1_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    assert datos["experimento"] == "E3"
    assert isinstance(datos["h1"], bool)
    assert len(datos["por_alpha"]) == 3
    for entrada in datos["por_alpha"]:
        assert "tau" in entrada
        assert "riesgo_empirico" in entrada
        assert "tasa_condicional" in entrada
        assert "llamadas_llm" in entrada
        assert "exito" in entrada


def test_g_dos_rapidas_iguales_salvo_fecha(tmp_path):
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    e3.main(["--rapido", "--modo", "simulada", "--out", str(out1), "--e1", str(_e1_repo())])
    e3.main(["--rapido", "--modo", "simulada", "--out", str(out2), "--e1", str(_e1_repo())])
    d1 = json.loads(out1.read_text(encoding="utf-8"))
    d2 = json.loads(out2.read_text(encoding="utf-8"))
    d1.pop("fecha_utc")
    d2.pop("fecha_utc")
    assert d1 == d2


def test_r1_alpha_cero_acepta_nada(tmp_path):
    out = tmp_path / "E3.json"
    e3.main(["--rapido", "--modo", "simulada", "--alphas", "0.0,0.05", "--out", str(out), "--e1", str(_e1_repo())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    entrada = [e for e in datos["por_alpha"] if abs(float(e["alpha"]) - 0.0) < 1e-12][0]
    assert all(v is None for v in entrada["lambda"])
    assert all(v is None for v in entrada["tau"])
    assert all(bool(v) for v in entrada["acepta_nada"])
    assert float(entrada["riesgo_empirico"]) == 0.0
    assert float(entrada["llamadas_llm"]) == float(entrada["pasos"])
    assert entrada["tasa_condicional"] is None


def test_r2_truncado_replay_sin_cache(tmp_path):
    out = tmp_path / "E3.json"
    cache_vacia = tmp_path / "cache_vacia"

    class _FallaSiUsa:
        def post(self, url, cabeceras, cuerpo):
            raise AssertionError("no debe tocar red en replay truncado")

    e3.main(
        ["--rapido", "--modo", "replay", "--max-llm-calls", "0", "--cache", str(cache_vacia),
         "--out", str(out), "--e1", str(_e1_repo())],
        transporte=_FallaSiUsa(),
    )
    datos = json.loads(out.read_text(encoding="utf-8"))
    validar(datos, _schema())
    assert datos["truncada"] is True
    assert datos["h1"] is False
