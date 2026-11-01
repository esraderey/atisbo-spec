"""Cierre O7: resumen derivado, README PLIEGO 10 y artefactos. Sin red."""
from __future__ import annotations

import importlib
import json
import re
import shlex
import tarfile
import tomllib
from pathlib import Path

import pytest

import demo.video_lazo as video_mod
from demo.registro import estado_a_dict
from atisbo.estado import EstadoSim
from atisbo.env.trazas import CUBOS_FIJOS

RD = Path("results")
README = Path("README.md")
TAR = Path("replay/cache.tar.xz")
PYPROJECT = Path("pyproject.toml")

LITERALES = [
    "muse-spark-1.3",
    "sam-3.1",
    "SAM 3.1",
    "12 × 12",
    "H1",
    "27 oct",
    "4 nov 2026",
    "Windows 11",
    "Python 3.13.2",
    "torch 2.15.0+cpu",
    "numpy 2.5.4",
]

NUM_RE = re.compile(r"(?<![\w.\-/])(?:\d{1,3}(?: \d{3})+|\d+)(?:\.\d+)?")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
CODEBLOCK_RE = re.compile(r"(```|~~~).*?\1", re.DOTALL)
INLINE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"https?://\S+")
CMD_RE = re.compile(r"-m\s+atisbo\.exp\.(e[1-5])([^`\n]*)")


def _base():
    return EstadoSim(
        bot=(6, 6),
        carga=None,
        objetos=((1, "rojo", 5, 5, False),),
        cubos=dict(CUBOS_FIJOS),
        bloqueo=None,
        paso=0,
    )


def _cargar_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _numeros_json():
    nums: list = []

    def _walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            nums.append(float(x))
            nums.append(x)
        elif isinstance(x, float):
            nums.append(x)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                _walk(v)

    for n in ["E1.json", "E2.json", "E2_sam.json", "E3.json", "E4.json", "E5.json", "resumen.json"]:
        _walk(_cargar_json(RD / n))
    return nums


def _miles(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def _seccion(texto: str, titulo: str) -> str:
    idx = texto.index(titulo)
    nxt = texto.find("\n## ", idx + 1)
    return texto[idx:] if nxt == -1 else texto[idx:nxt]


def _celda_por_nombre(celdas, nombre: str):
    for c in celdas:
        if str(c.get("nombre")) == nombre:
            return c
    raise AssertionError(f"celda {nombre} no encontrada")


def _celda_por_p(celdas, p: float):
    for c in celdas:
        if abs(float(c["config"]["p"]) - float(p)) < 1e-12:
            return c
    raise AssertionError(f"celda p={p} no encontrada")


def _episodios_celda(c):
    if c.get("n_episodios") is not None:
        return int(c["n_episodios"])
    return int(c["metricas"]["llamadas_llm"]["n"])


def test_a_estado_final_ultima_llamada(tmp_path, monkeypatch):
    s = estado_a_dict(_base())
    pasos = []
    for i in range(3):
        fuente = "atisbo" if i % 2 == 0 else "llm"
        pasos.append({"estado": s, "fuente": fuente, "accion": "DER", "conf": 0.9, "sorpresa": 0.0, "predichas": ["DER"]})
    final = estado_a_dict(EstadoSim(bot=(1, 2), carga=None, objetos=((1, "rojo", 5, 5, False),), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=3))
    reg = {"seed": 0, "episodio": 0, "tau": 0.5, "epsilon": 1.0, "pasos": pasos, "estado_final": final}
    jp = tmp_path / "reg.json"
    jp.write_text(json.dumps(reg), encoding="utf-8")
    out = tmp_path / "lazo.mp4"
    llamadas = []
    orig = video_mod.pintar_paso

    def espia(paso, n_llm, n_pasos):
        llamadas.append((dict(paso), int(n_llm), int(n_pasos)))
        return orig(paso, n_llm, n_pasos)

    monkeypatch.setattr(video_mod, "pintar_paso", espia)
    video_mod.main(["--episodio", str(jp), "--out", str(out)])
    assert len(llamadas) == len(pasos) + 1
    conteo = video_mod.contadores(pasos)
    ultimo_paso, ultimo_llm, ultimo_n = llamadas[-1]
    assert list(ultimo_paso["predichas"]) == []
    assert dict(ultimo_paso["estado"]) == dict(final)
    assert (ultimo_llm, ultimo_n) == (int(conteo[-1][0]), int(conteo[-1][1]))


def test_b_resumen_igual_y_totales():
    from demo.resumen import resumen

    got = resumen(RD)
    esperado = _cargar_json(RD / "resumen.json")
    assert got == esperado
    e1 = _cargar_json(RD / "E1.json")
    e2 = _cargar_json(RD / "E2.json")
    e2s = _cargar_json(RD / "E2_sam.json")
    e3 = _cargar_json(RD / "E3.json")
    e4 = _cargar_json(RD / "E4.json")
    e5 = _cargar_json(RD / "E5.json")
    suma_ll = int(e1["llamadas_llm"]) + int(e2["llamadas_llm"]) + int(e2s["llamadas_llm"]) + int(e3["llamadas_llm"]) + int(e4["llm"]["llamadas"]) + int(e5["llm"]["llamadas"])
    assert int(got["coste"]["llamadas_total"]) == suma_ll
    suma_coste = float(e1["coste_estimado_usd"]) + float(e2["coste_estimado_usd"]) + float(e2s["coste_estimado_usd"]) + float(e3["coste_estimado_usd"]) + float(e4["llm"]["coste_estimado_usd"]) + float(e5["llm"]["coste_estimado_usd"])
    assert float(got["coste"]["coste_total_usd"]) == pytest.approx(suma_coste, rel=1e-9, abs=1e-9)
    ratio = float(e2["h1_detalle"]["ratio_llamadas"])
    assert float(got["h1"]["e2"]["ahorro_pct"]) == pytest.approx(100.0 * (1.0 - ratio), rel=1e-9, abs=1e-9)


def test_c_readme_cabeceras_tablas_imagenes_cifras():
    from demo.resumen import tablas

    texto = README.read_text(encoding="utf-8")
    for h in ["## Built during the hackathon", "## How Meta models are used", "## Resultados", "## Limitaciones y resultados negativos", "## Reproducción", "## Coste de la evaluación"]:
        assert h in texto
    t = tablas(RD)
    partes = t.split("\n\n")
    assert len(partes) == 3
    for p in partes:
        assert p in texto
    for m in IMG_RE.finditer(texto):
        ruta = m.group(1).strip()
        assert Path(ruta).exists(), ruta
    sin_img = IMG_RE.sub(" ", texto)
    sin_bloques = CODEBLOCK_RE.sub(" ", sin_img)
    sin_linea = INLINE_RE.sub(" ", sin_bloques)
    sin_url = URL_RE.sub(" ", sin_linea)
    for lit in LITERALES:
        sin_url = sin_url.replace(lit, " ")
    pool = _numeros_json()
    for m in NUM_RE.finditer(sin_url):
        cifra = m.group(0).strip()
        if not cifra:
            continue
        norm = cifra.replace(" ", "").replace(" ", "")
        norm = norm.rstrip(".")
        if not norm or not norm[0].isdigit():
            continue
        d = len(norm.split(".")[1]) if "." in norm else 0
        ok = any(format(abs(float(n)), f".{d}f") == norm for n in pool)
        assert ok, cifra


def test_d_readme_comandos_parsean():
    texto = README.read_text(encoding="utf-8")
    matches = list(CMD_RE.finditer(texto))
    assert matches
    for m in matches:
        mod = f"atisbo.exp.{m.group(1)}"
        args = shlex.split(m.group(2).strip()) if m.group(2).strip() else []
        importlib.import_module(mod)._parse_args(list(args))
    sec = _seccion(texto, "## Reproducción")
    sec_matches = list(CMD_RE.finditer(sec))
    assert len(sec_matches) >= 6
    for m in sec_matches:
        args = shlex.split(m.group(2).strip()) if m.group(2).strip() else []
        assert "--modo" in args
        assert args[args.index("--modo") + 1] == "replay"


def test_e_cache_tar():
    assert TAR.exists()
    with tarfile.open(str(TAR), mode="r:xz") as tf:
        miembros = tf.getmembers()
    n_arch = 0
    for mi in miembros:
        nombre = mi.name.replace("\\", "/")
        assert nombre == "cache/llm" or nombre == "cache/sam" or nombre.startswith("cache/llm/") or nombre.startswith("cache/sam/"), nombre
        if mi.isfile():
            n_arch += 1
    assert n_arch > 16000
    with tarfile.open(str(TAR), mode="r:xz") as tf:
        for mi in tf.getmembers():
            if not mi.isfile():
                continue
            f = tf.extractfile(mi)
            assert f is not None
            data = f.read().decode("utf-8", errors="ignore").lower()
            for s in ["bearer", "authorization", "api_key", "x-api-key"]:
                assert s not in data, mi.name


def test_f_pyproject_extras():
    datos = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    opt = datos["project"]["optional-dependencies"]
    assert "demo" in opt
    assert any("matplotlib" in str(x).lower() for x in opt["demo"])
    assert "test" in opt
    assert any("matplotlib" in str(x).lower() for x in opt["test"])


def test_g_filas_tablas_contra_json():
    from demo.resumen import tablas

    t = tablas(RD)
    e1 = _cargar_json(RD / "E1.json")
    e2 = _cargar_json(RD / "E2.json")
    e2s = _cargar_json(RD / "E2_sam.json")
    e3 = _cargar_json(RD / "E3.json")
    e4 = _cargar_json(RD / "E4.json")
    e5 = _cargar_json(RD / "E5.json")

    def fila_resultados(etiqueta, episodios, met_ll, met_ex):
        ll = float(met_ll["media"])
        lo, hi = met_ll["ic95"]
        ex = float(met_ex["media"])
        elo, ehi = met_ex["ic95"]
        return f"| {etiqueta} | {int(episodios)} | {ll:.2f} | {float(lo):.2f}–{float(hi):.2f} | {ex:.2f} | {float(elo):.2f}–{float(ehi):.2f} |"

    e2_ep = int(e2["config"]["episodios_completados"])
    exp_e2 = fila_resultados("E2: Atisbo, τ = 0.5", e2_ep, e2["metricas"]["llamadas_llm"], e2["metricas"]["exito"])
    assert exp_e2 in t
    e2s_ep = int(e2s["config"]["episodios_completados"])
    exp_e2s = fila_resultados("E2_sam: E2 con estado desde SAM 3.1", e2s_ep, e2s["metricas"]["llamadas_llm"], e2s["metricas"]["exito"])
    assert exp_e2s in t
    ent05 = next(e for e in e3["por_alpha"] if abs(float(e["alpha"]) - 0.05) < 1e-12)
    exp_e305 = fila_resultados("E3: CRC, α = 0.05", int(ent05["n_episodios"]), e3["metricas"]["llamadas_llm@0.05"], e3["metricas"]["exito@0.05"])
    assert exp_e305 in t
    c_h1 = _celda_por_nombre(e4["celdas"], "horizonte_1")
    exp_e4h1 = fila_resultados("E4: horizonte 1", _episodios_celda(c_h1), c_h1["metricas"]["llamadas_llm"], c_h1["metricas"]["exito"])
    assert exp_e4h1 in t
    c_p005 = _celda_por_p(e5["celdas"], 0.05)
    exp_e5 = fila_resultados("E5: p = 0.05", _episodios_celda(c_p005), c_p005["metricas"]["llamadas_llm"], c_p005["metricas"]["exito"])
    assert exp_e5 in t

    ent01 = next(e for e in e3["por_alpha"] if abs(float(e["alpha"]) - 0.1) < 1e-12)
    taus = [float(v) for v in ent01["tau"] if v is not None]
    tmin, tmax = min(taus), max(taus)
    rt = e3["metricas"]["riesgo_trazas@0.1"]
    re_ = e3["metricas"]["riesgo_empirico@0.1"]
    rt_lo, rt_hi = rt["ic95"]
    re_lo, re_hi = re_["ic95"]
    exp_riesgo = f"| 0.10 | {tmin:.2f}–{tmax:.2f} | {float(rt['media']):.4f} | {float(rt_lo):.4f}–{float(rt_hi):.4f} | {float(re_['media']):.3f} | {float(re_lo):.3f}–{float(re_hi):.3f} |"
    assert exp_riesgo in t

    media_e1 = float(e1["metricas"]["llamadas_llm"]["media"])

    def fila_coste(etiqueta, episodios, llamadas, por_txt, ex_txt, coste_txt, sin_txt):
        return f"| {etiqueta} | {int(episodios)} | {int(llamadas)} | {por_txt} | {ex_txt} | {coste_txt} | {sin_txt} |"

    sin_e2s = int(round(float(e2s_ep) * media_e1, -2))
    exp_c_e2s = fila_coste(
        "E2_sam: E2 con estado desde SAM 3.1",
        e2s_ep,
        int(e2s["llamadas_llm"]),
        f"{float(e2s['metricas']['llamadas_llm']['media']):.2f}",
        f"{float(e2s['metricas']['exito']['media']):.2f}",
        f"{float(e2s['coste_estimado_usd']):.2f} $",
        f"≈ {_miles(sin_e2s)}",
    )
    assert exp_c_e2s in t
    e4_ep = int(sum(_episodios_celda(c) for c in e4["celdas"]))
    sin_e4 = int(round(float(e4_ep) * media_e1, -2))
    exp_c_e4 = fila_coste(
        "E4 en total",
        e4_ep,
        int(e4["llm"]["llamadas"]),
        "",
        "",
        f"{float(e4['llm']['coste_estimado_usd']):.2f} $",
        f"≈ {_miles(sin_e4)}",
    )
    assert exp_c_e4 in t
