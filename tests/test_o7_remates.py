"""Remates O7: cada frase del README contra los JSON, frase a frase. Sin red."""
from __future__ import annotations

import importlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RD = Path("results")
README_PATH = Path("README.md")
CMD_ATISBO_RE = re.compile(r"-m\s+atisbo\.exp\.(e[1-5])([^\n`~]*)")
CMD_DEMO_RE = re.compile(r"-m\s+(demo\.[A-Za-z_]+)([^\n`~]*)")
CMD_TAR_RE = re.compile(r"-m\s+tarfile\s+-e\s+(\S+)")

PARES_RESULTADOS = [
    ("E1: Muse Spark decide cada paso", "E1"),
    ("E2: Atisbo, τ = 0.5", "E2"),
    ("E2_sam: E2 con estado desde SAM 3.1", "E2_sam"),
    ("E3: CRC, α = 0.01", "E3@0.01"),
    ("E3: CRC, α = 0.05", "E3@0.05"),
    ("E3: CRC, α = 0.1", "E3@0.1"),
    ("E4: sin sorpresa", "E4:sin_sorpresa"),
    ("E4: horizonte 1", "E4:horizonte_1"),
    ("E5: p = 0", "E5@p0"),
    ("E5: p = 0.05", "E5@p005"),
]

PARES_RIESGO = [
    ("0.01", 0.01),
    ("0.05", 0.05),
    ("0.1", 0.1),
]

PARES_COSTE = [
    ("E1: Muse Spark decide cada paso", "E1"),
    ("E2: Atisbo, τ = 0.5", "E2"),
    ("E2_sam: E2 con estado desde SAM 3.1", "E2_sam"),
    ("E3: CRC, α = 0.01", "E3@0.01"),
    ("E3: CRC, α = 0.05", "E3@0.05"),
    ("E3: CRC, α = 0.1", "E3@0.1"),
    ("E3 en total", "E3-total"),
    ("E4 en total", "E4-total"),
    ("E5 en total", "E5-total"),
]


def _cargar(nombre: str) -> dict:
    return json.loads((RD / nombre).read_text(encoding="utf-8"))


def _miles(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def _por_alpha(entries, alpha: float) -> dict:
    for e in entries:
        if abs(float(e["alpha"]) - float(alpha)) < 1e-12:
            return e
    raise AssertionError(f"alpha {alpha} no encontrado")


def _celda_por_nombre(celdas, nombre: str) -> dict:
    for c in celdas:
        if str(c.get("nombre")) == nombre:
            return c
    raise AssertionError(f"celda {nombre} no encontrada")


def _celda_por_p(celdas, p: float) -> dict:
    for c in celdas:
        if abs(float(c["config"]["p"]) - float(p)) < 1e-12:
            return c
    raise AssertionError(f"celda p={p} no encontrada")


def _fila_resultados_esperada(etiqueta: str, episodios: int, met_ll: dict, met_ex: dict) -> str:
    ll = float(met_ll["media"])
    lo, hi = met_ll["ic95"]
    ex = float(met_ex["media"])
    elo, ehi = met_ex["ic95"]
    return f"| {etiqueta} | {int(episodios)} | {ll:.2f} | {float(lo):.2f}–{float(hi):.2f} | {ex:.2f} | {float(elo):.2f}–{float(ehi):.2f} |"


def _fila_riesgo_esperada(entrada: dict, metricas: dict) -> str:
    a = float(entrada["alpha"])
    taus = [float(v) for v in list(entrada["tau"]) if v is not None]
    tmin = float(min(taus))
    tmax = float(max(taus))
    rt = metricas[f"riesgo_trazas@{float(a)}"]
    re_ = metricas[f"riesgo_empirico@{float(a)}"]
    rt_lo, rt_hi = rt["ic95"]
    re_lo, re_hi = re_["ic95"]
    return f"| {float(a):.2f} | {tmin:.2f}–{tmax:.2f} | {float(rt['media']):.4f} | {float(rt_lo):.4f}–{float(rt_hi):.4f} | {float(re_['media']):.3f} | {float(re_lo):.3f}–{float(re_hi):.3f} |"


def _fila_coste_esperada(experimento: str, episodios: int, llamadas: int, por_txt: str, ex_txt: str, coste_txt: str, sin_txt: str) -> str:
    return f"| {experimento} | {int(episodios)} | {int(llamadas)} | {por_txt} | {ex_txt} | {coste_txt} | {sin_txt} |"


def _esperada_resultado_para(origen: str) -> str:
    e1 = _cargar("E1.json")
    e2 = _cargar("E2.json")
    e2s = _cargar("E2_sam.json")
    e3 = _cargar("E3.json")
    e4 = _cargar("E4.json")
    e5 = _cargar("E5.json")
    if origen == "E1":
        return _fila_resultados_esperada("E1: Muse Spark decide cada paso", int(e1["config"]["episodios_completados"]), e1["metricas"]["llamadas_llm"], e1["metricas"]["exito"])
    if origen == "E2":
        return _fila_resultados_esperada("E2: Atisbo, τ = 0.5", int(e2["config"]["episodios_completados"]), e2["metricas"]["llamadas_llm"], e2["metricas"]["exito"])
    if origen == "E2_sam":
        return _fila_resultados_esperada("E2_sam: E2 con estado desde SAM 3.1", int(e2s["config"]["episodios_completados"]), e2s["metricas"]["llamadas_llm"], e2s["metricas"]["exito"])
    if origen.startswith("E3@"):
        a = float(origen.split("@")[1])
        ent = _por_alpha(e3["por_alpha"], a)
        etiqueta = "E3: CRC, α = 0.1" if a == 0.1 else f"E3: CRC, α = {a:.2f}"
        return _fila_resultados_esperada(etiqueta, int(ent["n_episodios"]), e3["metricas"][f"llamadas_llm@{float(a)}"], e3["metricas"][f"exito@{float(a)}"])
    if origen == "E4:sin_sorpresa":
        c = _celda_por_nombre(e4["celdas"], "sin_sorpresa")
        return _fila_resultados_esperada("E4: sin sorpresa", int(c["n_episodios"]), c["metricas"]["llamadas_llm"], c["metricas"]["exito"])
    if origen == "E4:horizonte_1":
        c = _celda_por_nombre(e4["celdas"], "horizonte_1")
        return _fila_resultados_esperada("E4: horizonte 1", int(c["n_episodios"]), c["metricas"]["llamadas_llm"], c["metricas"]["exito"])
    if origen == "E5@p0":
        c = _celda_por_p(e5["celdas"], 0.0)
        return _fila_resultados_esperada("E5: p = 0", int(c["n_episodios"]), c["metricas"]["llamadas_llm"], c["metricas"]["exito"])
    if origen == "E5@p005":
        c = _celda_por_p(e5["celdas"], 0.05)
        return _fila_resultados_esperada("E5: p = 0.05", int(c["n_episodios"]), c["metricas"]["llamadas_llm"], c["metricas"]["exito"])
    raise AssertionError(f"origen {origen} desconocido")


def _esperada_coste_para(origen: str) -> str:
    e1 = _cargar("E1.json")
    e2 = _cargar("E2.json")
    e2s = _cargar("E2_sam.json")
    e3 = _cargar("E3.json")
    e4 = _cargar("E4.json")
    e5 = _cargar("E5.json")
    media_e1 = float(e1["metricas"]["llamadas_llm"]["media"])

    def _sin(episodios: int) -> str:
        return f"≈ {_miles(int(round(float(episodios) * media_e1, -2)))}"

    if origen == "E1":
        return _fila_coste_esperada("E1: Muse Spark decide cada paso", int(e1["config"]["episodios_completados"]), int(e1["llamadas_llm"]), f"{float(e1['metricas']['llamadas_llm']['media']):.2f}", f"{float(e1['metricas']['exito']['media']):.2f}", f"{float(e1['coste_estimado_usd']):.2f} $", "—")
    if origen == "E2":
        return _fila_coste_esperada("E2: Atisbo, τ = 0.5", int(e2["config"]["episodios_completados"]), int(e2["llamadas_llm"]), f"{float(e2['metricas']['llamadas_llm']['media']):.2f}", f"{float(e2['metricas']['exito']['media']):.2f}", f"{float(e2['coste_estimado_usd']):.2f} $", _sin(int(e2["config"]["episodios_completados"])))
    if origen == "E2_sam":
        return _fila_coste_esperada("E2_sam: E2 con estado desde SAM 3.1", int(e2s["config"]["episodios_completados"]), int(e2s["llamadas_llm"]), f"{float(e2s['metricas']['llamadas_llm']['media']):.2f}", f"{float(e2s['metricas']['exito']['media']):.2f}", f"{float(e2s['coste_estimado_usd']):.2f} $", _sin(int(e2s["config"]["episodios_completados"])))
    if origen.startswith("E3@"):
        a = float(origen.split("@")[1])
        ent = _por_alpha(e3["por_alpha"], a)
        n_ep = int(ent["n_episodios"])
        met_ll = e3["metricas"][f"llamadas_llm@{float(a)}"]
        met_ex = e3["metricas"][f"exito@{float(a)}"]
        llamadas = int(round(float(met_ll["media"]) * float(n_ep)))
        etiqueta = "E3: CRC, α = 0.1" if a == 0.1 else f"E3: CRC, α = {a:.2f}"
        return _fila_coste_esperada(etiqueta, n_ep, llamadas, f"{float(met_ll['media']):.2f}", f"{float(met_ex['media']):.2f}", "—", _sin(n_ep))
    if origen == "E3-total":
        e3_ep = int(sum(int(e["n_episodios"]) for e in e3["por_alpha"]))
        return _fila_coste_esperada("E3 en total", e3_ep, int(e3["llamadas_llm"]), "", "", f"{float(e3['coste_estimado_usd']):.2f} $", _sin(e3_ep))
    if origen == "E4-total":
        e4_ep = int(sum(int(c["n_episodios"]) for c in e4["celdas"]))
        return _fila_coste_esperada("E4 en total", e4_ep, int(e4["llm"]["llamadas"]), "", "", f"{float(e4['llm']['coste_estimado_usd']):.2f} $", _sin(e4_ep))
    if origen == "E5-total":
        e5_ep = int(sum(int(c["n_episodios"]) for c in e5["celdas"]))
        return _fila_coste_esperada("E5 en total", e5_ep, int(e5["llm"]["llamadas"]), "", "", f"{float(e5['llm']['coste_estimado_usd']):.2f} $", _sin(e5_ep))
    raise AssertionError(f"origen coste {origen} desconocido")


def _seccion(texto: str, titulo: str) -> str:
    idx = texto.index(titulo)
    nxt = texto.find("\n## ", idx + 1)
    return texto[idx:] if nxt == -1 else texto[idx:nxt]


def test_a_resumen_stdout_utf8_y_json(tmp_path):
    out = tmp_path / "resumen.json"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    proc = subprocess.run(
        [sys.executable, "-m", "demo.resumen", "--results", "results", "--out", str(out)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert proc.returncode == 0
    texto = proc.stdout.decode("utf-8")
    assert "τ" in texto
    got = json.loads(out.read_text(encoding="utf-8"))
    esperado = json.loads((RD / "resumen.json").read_text(encoding="utf-8"))
    assert got == esperado


def test_b_filas_exactas_contra_json():
    texto = README_PATH.read_text(encoding="utf-8")
    assert len(PARES_RESULTADOS) == 10
    for etiqueta, origen in PARES_RESULTADOS:
        esperada = _esperada_resultado_para(origen)
        assert esperada in texto
    e3 = _cargar("E3.json")
    assert len(PARES_RIESGO) == 3
    for _etiqueta, alpha in PARES_RIESGO:
        ent = _por_alpha(e3["por_alpha"], float(alpha))
        esperada = _fila_riesgo_esperada(ent, e3["metricas"])
        assert esperada in texto
    assert len(PARES_COSTE) == 9
    for _etiqueta, origen in PARES_COSTE:
        esperada = _esperada_coste_para(origen)
        assert esperada in texto


def test_c_derivado_campo_a_campo():
    e1 = _cargar("E1.json")
    e2 = _cargar("E2.json")
    e2s = _cargar("E2_sam.json")
    e3 = _cargar("E3.json")
    e4 = _cargar("E4.json")
    e5 = _cargar("E5.json")
    reg = _cargar("registro_E2_seed0.json")
    der = _cargar("resumen.json")
    for clave, src in [("e2", e2), ("e2_sam", e2s), ("e3", e3)]:
        ratio = float(src["h1_detalle"]["ratio_llamadas"])
        assert float(der["h1"][clave]["ratio_llamadas"]) == pytest.approx(ratio, rel=1e-9, abs=1e-9)
        assert float(der["h1"][clave]["ahorro_pct"]) == pytest.approx(100.0 * (1.0 - ratio), rel=1e-9, abs=1e-9)
        assert float(der["h1"][clave]["delta_exito_pp"]) == pytest.approx(float(src["h1_detalle"]["delta_exito_pp"]), rel=1e-9, abs=1e-9)
    for a in [0.01, 0.05, 0.1]:
        ent = _por_alpha(e3["por_alpha"], float(a))
        got = der["riesgo"][str(float(a))]
        assert float(got["alpha"]) == pytest.approx(float(a), rel=0, abs=0)
        assert float(got["riesgo_empirico"]) == pytest.approx(float(ent["riesgo_empirico"]), rel=1e-9, abs=1e-9)
        assert float(got["veces"]) == pytest.approx(float(ent["riesgo_empirico"]) / float(a), rel=1e-9, abs=1e-9)
    media_e1 = float(e1["metricas"]["llamadas_llm"]["media"])
    met_a001 = e3["metricas"]["llamadas_llm@0.01"]
    assert float(der["ahorro_alpha_001_pct"]) == pytest.approx(100.0 * (1.0 - float(met_a001["media"]) / media_e1), rel=1e-9, abs=1e-9)
    assert float(der["sam"]["coste_llm_usd"]) == pytest.approx(float(e2s["llm"]["coste_estimado_usd"]), rel=1e-9, abs=1e-9)
    assert float(der["sam"]["coste_sam_usd"]) == pytest.approx(float(e2s["sam"]["coste_estimado_usd"]), rel=1e-9, abs=1e-9)
    assert int(der["sam"]["imagenes"]) == int(e2s["sam"]["imagenes"])
    assert float(der["sam"]["coincidencia_bot_pct"]) == pytest.approx(float(e2s["metricas"]["coincidencia_bot"]["media"]) * 100.0, rel=1e-9, abs=1e-9)
    assert float(der["sam"]["coincidencia_todo_pct"]) == pytest.approx(float(e2s["metricas"]["coincidencia_todo"]["media"]) * 100.0, rel=1e-9, abs=1e-9)
    assert int(der["video"]["episodio"]) == int(reg["episodio"])
    assert int(der["video"]["pasos"]) == int(len(list(reg["pasos"])))
    assert int(der["video"]["llamadas_llm"]) == int(reg["llamadas_llm"])
    assert float(der["video"]["pct_llm"]) == pytest.approx(100.0 * float(reg["llamadas_llm"]) / float(len(list(reg["pasos"]))), rel=1e-9, abs=1e-9)
    assert int(der["coste"]["sin_atisbo_e2_e3"]) == int(round(200.0 * media_e1, -2))
    assert int(der["coste"]["con_atisbo_e2_e3"]) == int(int(e2["llamadas_llm"]) + int(e3["llamadas_llm"]))
    assert int(der["coste"]["llamadas_total"]) == int(int(e1["llamadas_llm"]) + int(e2["llamadas_llm"]) + int(e2s["llamadas_llm"]) + int(e3["llamadas_llm"]) + int(e4["llm"]["llamadas"]) + int(e5["llm"]["llamadas"]))


def test_d_frases_literales_desde_json():
    texto = README_PATH.read_text(encoding="utf-8")
    e1 = _cargar("E1.json")
    e2 = _cargar("E2.json")
    e2s = _cargar("E2_sam.json")
    e3 = _cargar("E3.json")
    e4 = _cargar("E4.json")
    e5 = _cargar("E5.json")
    der = _cargar("resumen.json")
    e1m = float(e1["metricas"]["llamadas_llm"]["media"])
    e2m = float(e2["metricas"]["llamadas_llm"]["media"])
    ex1 = float(e1["metricas"]["exito"]["media"])
    ex2 = float(e2["metricas"]["exito"]["media"])
    assert f"E2 baja de {e1m:.2f} a {e2m:.2f} llamadas por episodio con éxito {ex2:.2f} frente a {ex1:.2f}" in texto
    ahorro = float(der["h1"]["e2"]["ahorro_pct"])
    assert f"E2 ahorra el {ahorro:.1f} % de las llamadas" in texto
    ahorro_001 = float(der["ahorro_alpha_001_pct"])
    assert f"se ahorra el {ahorro_001:.1f} %" in texto
    veces = [float(der["riesgo"][str(float(a))]["veces"]) for a in [0.01, 0.05, 0.1]]
    assert f"{veces[0]:.1f}, {veces[1]:.1f} y {veces[2]:.1f} veces α" in texto
    for clave in ["e2", "e2_sam", "e3"]:
        r = float(der["h1"][clave]["ratio_llamadas"])
        d = float(der["h1"][clave]["delta_exito_pp"])
        assert f"{r:.3f} y −{abs(d):.0f} pp" in texto
    pasos = int(der["video"]["pasos"])
    llamadas = int(der["video"]["llamadas_llm"])
    assert f"con {pasos} pasos y {llamadas} llamadas" in texto
    sam_c = float(der["sam"]["coste_sam_usd"])
    llm_c = float(der["sam"]["coste_llm_usd"])
    assert f"{sam_c:.2f} $ frente a {llm_c:.2f} $" in texto
    total = int(der["coste"]["llamadas_total"])
    assert f"suman {_miles(total)} llamadas al LLM" in texto
    imagenes = int(der["sam"]["imagenes"])
    assert f"En total son {imagenes} imágenes" in texto
    coste = float(der["coste"]["coste_total_usd"])
    assert f"{_miles(total)} llamadas y {coste:.2f} $ estimados" in texto
    r3 = int(e3["llm"]["llamadas_red"])
    r4 = int(e4["llm"]["llamadas_red"])
    r5 = int(e5["llm"]["llamadas_red"])
    assert f"de las {int(e3['llamadas_llm'])}, {int(e4['llm']['llamadas'])} y {int(e5['llm']['llamadas'])} llamadas de E3, E4 y E5 fueron a la red {r3}, {r4} y {r5}" in texto
    red_sam = int(e2s["sam"]["llamadas_red"])
    assert f"fueron a la red {red_sam} de las {imagenes} imágenes" in texto
    sorpresa = float(e2["metricas"]["llamadas_por_sorpresa"]["media"])
    assert f"{sorpresa:.2f} llamadas por sorpresa" in texto
    sin_c = _celda_por_nombre(e4["celdas"], "sin_sorpresa")
    base_c = _celda_por_nombre(e4["celdas"], "base")
    a_seed = float(sin_c["metricas"]["llamadas_llm"]["por_seed"][3])
    b_seed = float(base_c["metricas"]["llamadas_llm"]["por_seed"][3])
    assert f"{a_seed:.2f} frente a {b_seed:.2f} en la semilla 3" in texto
    p005 = _celda_por_nombre(e5["celdas"], "p005")
    x = float(p005["metricas"]["llamadas_por_sorpresa"]["media"])
    assert f"salta {x:.2f} veces por episodio" in texto
    n_fallos = int(round((1.0 - float(e2["metricas"]["exito"]["media"])) * int(e2["config"]["episodios_completados"])))
    m_eps = int(e2["config"]["episodios_completados"])
    assert f"E2 falla {n_fallos} de {m_eps}" in texto
    sin_e2e3 = int(der["coste"]["sin_atisbo_e2_e3"])
    con_e2e3 = int(der["coste"]["con_atisbo_e2_e3"])
    assert f"unas {_miles(sin_e2e3)} llamadas; con Atisbo fueron {con_e2e3}" in texto


def test_e_sin_codigo_solo_cifras():
    texto = README_PATH.read_text(encoding="utf-8")
    for s in re.findall(r"`([^`\n]+)`", texto):
        assert not re.fullmatch(r"[\d\s.,]+", s), s


def test_f_reproduccion_orden_y_args():
    texto = README_PATH.read_text(encoding="utf-8")
    sec = _seccion(texto, "## Reproducción")
    matches = list(CMD_ATISBO_RE.finditer(sec))
    assert len(matches) == 6
    mods = [m.group(1) for m in matches]
    assert mods == ["e1", "e2", "e2", "e3", "e4", "e5"]
    args_list = [shlex.split(m.group(2).strip()) if m.group(2).strip() else [] for m in matches]
    assert "--estado" not in args_list[1]
    assert "sam" in args_list[2]
    assert args_list[2][args_list[2].index("--estado") + 1] == "sam"
    e1 = _cargar("E1.json")
    e2 = _cargar("E2.json")
    e2s = _cargar("E2_sam.json")
    e3 = _cargar("E3.json")
    e4 = _cargar("E4.json")
    e5 = _cargar("E5.json")
    outs = ["results/E1.json", "results/E2.json", "results/E2_sam.json", "results/E3.json", "results/E4.json", "results/E5.json"]
    cfgs = [e1, e2, e2s, e3, e4, e5]
    for args, out, cfg in zip(args_list, outs, cfgs):
        assert "--modo" in args
        assert args[args.index("--modo") + 1] == "replay"
        assert "--out" in args
        assert args[args.index("--out") + 1] == out
        assert "--max-llm-calls" in args
        assert int(args[args.index("--max-llm-calls") + 1]) == int(cfg["config"]["max_llm_calls"])
    a1 = args_list[0]
    assert "--seed" in a1
    assert int(a1[a1.index("--seed") + 1]) == int(e1["config"]["seed"])
    assert "--n" in a1
    assert int(a1[a1.index("--n") + 1]) == int(e1["config"]["n_episodios"])
    assert "--p" in a1
    assert float(a1[a1.index("--p") + 1]) == pytest.approx(float(e1["config"]["p_perturb"]), rel=0, abs=1e-12)
    a_sam = args_list[2]
    assert "--seeds" in a_sam
    assert [int(x) for x in a_sam[a_sam.index("--seeds") + 1].split(",")] == [int(x) for x in e2s["config"]["seeds"]]
    pos_e4 = mods.index("e4")
    pos_e3 = mods.index("e3")
    pos_sam = next(i for i, args in enumerate(args_list) if "--estado" in args and args[args.index("--estado") + 1] == "sam")
    assert pos_e3 < pos_e4
    assert pos_sam < pos_e4
    for m in CMD_DEMO_RE.finditer(sec):
        mod = m.group(1)
        args = shlex.split(m.group(2).strip()) if m.group(2).strip() else []
        importlib.import_module(mod)._parse_args(list(args))
    m_tar = CMD_TAR_RE.search(sec)
    assert m_tar is not None
    assert Path(m_tar.group(1)).exists()


def test_g_hackathon_sin_mito():
    texto = README_PATH.read_text(encoding="utf-8")
    sec = _seccion(texto, "## Built during the hackathon")
    assert "Sin código previo" not in sec
    assert ".agents/" in sec
