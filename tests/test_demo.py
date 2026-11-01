"""Demo lazo y figuras (O7 parte 1): anclas rojas sin red, con tmp_path."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _estado_base():
    from atisbo.env.trazas import CUBOS_FIJOS
    from atisbo.estado import EstadoSim

    return EstadoSim(
        bot=(6, 6),
        carga=None,
        objetos=((1, "rojo", 5, 5, False), (2, "verde", 7, 7, False)),
        cubos=dict(CUBOS_FIJOS),
        bloqueo=None,
        paso=0,
    )


def _registro_12():
    from demo.registro import estado_a_dict

    base = _estado_base()
    pasos = []
    for i in range(12):
        from atisbo.estado import EstadoSim

        s = EstadoSim(
            bot=base.bot,
            carga=base.carga,
            objetos=base.objetos,
            cubos=dict(base.cubos),
            bloqueo=base.bloqueo,
            paso=i,
        )
        fuente = "atisbo" if i % 2 == 0 else "llm"
        pasos.append(
            {
                "estado": estado_a_dict(s),
                "fuente": fuente,
                "accion": "DER",
                "conf": 0.9 if fuente == "atisbo" else 0.1,
                "sorpresa": 0.0,
                "predichas": ["DER"],
            }
        )
    return pasos


def _resultados_minimos(d: Path, con_e5=True, con_sam=False):
    def _met(media, lo, hi):
        return {"media": float(media), "ic95": [float(lo), float(hi)], "n": 10, "n_remuestreos": 1000, "por_seed": [float(media)]}

    e1 = {
        "experimento": "E1",
        "fecha_utc": "2026-01-01T00:00:00Z",
        "commit": "x",
        "seeds": [0],
        "config": {},
        "metricas": {"llamadas_llm": _met(10.0, 9.0, 11.0), "exito": _met(0.8, 0.7, 0.9)},
        "exito_media": 0.8,
        "llm": {"modo": "simulada"},
        "notas": [],
    }
    e2 = {
        "experimento": "E2",
        "fecha_utc": "2026-01-01T00:00:00Z",
        "commit": "x",
        "seeds": [0],
        "config": {},
        "metricas": {"llamadas_llm": _met(3.0, 2.5, 3.5), "exito": _met(0.78, 0.68, 0.88)},
        "exito_media": 0.78,
        "llm": {"modo": "simulada"},
        "notas": [],
    }
    alphas = [0.01, 0.05, 0.10]
    llamadas = [5.0, 3.0, 2.0]
    rt = [0.005, 0.03, 0.07]
    re_ = [0.006, 0.028, 0.075]
    por_alpha = []
    metricas3 = {}
    for a, ll, r1, r2 in zip(alphas, llamadas, rt, re_):
        por_alpha.append(
            {"alpha": float(a), "llamadas_llm": float(ll), "riesgo_trazas": float(r1), "riesgo_empirico": float(r2)}
        )
        metricas3[f"llamadas_llm@{a}"] = _met(ll, ll - 0.5, ll + 0.5)
        metricas3[f"riesgo_trazas@{a}"] = _met(r1, max(0.0, r1 - 0.002), r1 + 0.002)
        metricas3[f"riesgo_empirico@{a}"] = _met(r2, max(0.0, r2 - 0.002), r2 + 0.002)
    e3 = {
        "experimento": "E3",
        "fecha_utc": "2026-01-01T00:00:00Z",
        "commit": "x",
        "seeds": [0],
        "config": {},
        "por_alpha": por_alpha,
        "metricas": metricas3,
        "notas": [],
    }
    (d / "E1.json").write_text(json.dumps(e1), encoding="utf-8")
    (d / "E2.json").write_text(json.dumps(e2), encoding="utf-8")
    (d / "E3.json").write_text(json.dumps(e3), encoding="utf-8")
    if con_sam:
        e2sam = {
            "experimento": "E2",
            "fecha_utc": "2026-01-01T00:00:00Z",
            "commit": "x",
            "seeds": [0],
            "config": {},
            "metricas": {"llamadas_llm": _met(4.0, 3.5, 4.5), "exito": _met(0.70, 0.60, 0.80)},
            "exito_media": 0.70,
            "llm": {"modo": "simulada"},
            "notas": [],
        }
        (d / "E2_sam.json").write_text(json.dumps(e2sam), encoding="utf-8")
    if con_e5:
        celdas = []
        for nombre, p, ll, ex in [("p0", 0.0, 2.5, 0.85), ("p002", 0.02, 3.0, 0.78), ("p005", 0.05, 4.0, 0.65)]:
            celdas.append(
                {
                    "nombre": nombre,
                    "config": {"p": float(p), "tau": 0.5, "sin_sorpresa": False, "horizonte": 3},
                    "n_episodios": 10,
                    "cache_miss": 0,
                    "seeds_evaluadas": [0],
                    "metricas": {"llamadas_llm": _met(ll, ll - 0.5, ll + 0.5), "exito": _met(ex, ex - 0.05, ex + 0.05)},
                }
            )
        e5 = {
            "experimento": "E5",
            "fecha_utc": "2026-01-01T00:00:00Z",
            "commit": "x",
            "seeds": [0],
            "config": {},
            "metricas": {},
            "celdas": celdas,
            "llm": {"modo": "simulada"},
            "notas": [],
            "truncada": False,
            "pasos_parcial": 0,
        }
        (d / "E5.json").write_text(json.dumps(e5), encoding="utf-8")


def test_video_mp4_12_frames_marcador(tmp_path):
    from demo.video_lazo import MARCADOR, main

    pasos = _registro_12()
    reg = {"seed": 0, "episodio": 0, "tau": 0.5, "epsilon": 1.0, "pasos": pasos}
    jp = tmp_path / "reg.json"
    jp.write_text(json.dumps(reg), encoding="utf-8")
    out = tmp_path / "lazo.mp4"
    main(["--episodio", str(jp), "--out", str(out)])
    assert out.exists() and out.stat().st_size > 1000
    with imageio.get_reader(str(out)) as lector:
        frames = [np.asarray(f) for f in lector]
    assert len(frames) == 12
    arr0 = np.asarray(frames[0])
    assert arr0.shape[0] == 384 + 64 and arr0.shape[1] == 384 and arr0.shape[2] == 3
    x0, y0, x1, y1 = MARCADOR
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    for i, fr in enumerate(frames):
        a = np.asarray(fr)
        assert a.shape == arr0.shape
        px = a[cy, cx].astype(int)
        if pasos[i]["fuente"] == "atisbo":
            assert px[1] > px[0] and px[1] > px[2]
        else:
            assert px[0] > px[1] and px[0] > px[2]


def test_contadores(tmp_path):
    from demo.video_lazo import contadores

    pasos = _registro_12()
    res = contadores(pasos)
    assert len(res) == 12
    n_llm = 0
    for i, (c_llm, c_pasos) in enumerate(res):
        if pasos[i]["fuente"] == "llm":
            n_llm += 1
        assert c_llm == n_llm
        assert c_pasos == i + 1


def test_texto_contador_cabe():
    from PIL import Image, ImageDraw

    from demo.video_lazo import ANCHO, _FUENTE

    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    assert d.textlength("llamadas LLM: 160 / pasos: 160", font=_FUENTE) <= ANCHO - 90


def test_fantasma_cambia_celdas():
    from demo.registro import estado_a_dict
    from demo.video_lazo import pintar_paso

    s = estado_a_dict(_estado_base())
    base = {"estado": s, "fuente": "atisbo", "accion": "DER", "conf": 0.9, "sorpresa": 0.0, "predichas": []}
    ghost = {"estado": s, "fuente": "atisbo", "accion": "DER", "conf": 0.9, "sorpresa": 0.0, "predichas": ["DER", "DER", "ABAJO"]}
    a = pintar_paso(base, 0, 1).astype(int)
    b = pintar_paso(ghost, 0, 1).astype(int)
    assert a.shape == (384 + 64, 384, 3)
    mundo_a = a[:384, :, :]
    mundo_b = b[:384, :, :]
    assert np.any(mundo_a != mundo_b)
    # Celdas esperadas desde (6,6): DER->(7,6), DER->(8,6), ABAJO->(8,7).
    for (cx, cy) in [(7, 6), (8, 6), (8, 7)]:
        px = cx * 32 + 16
        py = cy * 32 + 16
        assert np.any(mundo_a[py - 2:py + 3, px - 2:px + 3, :] != mundo_b[py - 2:py + 3, px - 2:px + 3, :])


def test_figuras_cuatro_png(tmp_path):
    from demo.figuras import main as fig_main

    resd = tmp_path / "results"
    outd = tmp_path / "out"
    resd.mkdir()
    _resultados_minimos(resd, con_e5=True, con_sam=False)
    fig_main(["--results", str(resd), "--out", str(outd)])
    for nombre in ["llamadas_vs_alpha.png", "riesgo_vs_alpha.png", "e1_vs_e2.png", "matriz_e5.png"]:
        p = outd / nombre
        assert p.exists() and p.stat().st_size > 1024


def test_figuras_sin_e5_falla(tmp_path):
    from demo.figuras import main as fig_main

    import pytest

    resd = tmp_path / "results"
    outd = tmp_path / "out"
    resd.mkdir()
    _resultados_minimos(resd, con_e5=False, con_sam=False)
    with pytest.raises(FileNotFoundError):
        fig_main(["--results", str(resd), "--out", str(outd)])


def test_figuras_sam_cambia(tmp_path):
    from demo.figuras import main as fig_main

    resd = tmp_path / "results"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    resd.mkdir()
    _resultados_minimos(resd, con_e5=True, con_sam=False)
    fig_main(["--results", str(resd), "--out", str(out1)])
    antes = (out1 / "e1_vs_e2.png").read_bytes()
    _resultados_minimos(resd, con_e5=True, con_sam=True)
    fig_main(["--results", str(resd), "--out", str(out2)])
    despues = (out2 / "e1_vs_e2.png").read_bytes()
    assert len(antes) > 1024 and len(despues) > 1024
    assert antes != despues


def test_capturar_nulo_igual_plain():
    from atisbo.control import PredictorNulo
    from atisbo.lazo import Mundo, correr_episodio
    from atisbo.politica_llm import PoliticaSimulada
    from demo.registro import capturar

    m1 = Mundo(seed=0, p_perturb=0.02)
    m2 = Mundo(seed=0, p_perturb=0.02)
    p1 = PoliticaSimulada(seed=0)
    p2 = PoliticaSimulada(seed=0)
    reg = []
    res = correr_episodio(PredictorNulo(), p1, m1, tau=0.5, epsilon=1e9, registro=reg)
    cap = capturar(PredictorNulo(), p2, m2, tau=0.5, epsilon=1e9)
    assert len(cap["pasos"]) == len(reg) > 0
    assert all(p["fuente"] == "llm" for p in cap["pasos"])
    assert cap["exito"] == res["exito"]
    assert cap["llamadas_llm"] == res["llamadas_llm"] == len(reg)
    assert [p["accion"] for p in cap["pasos"]] == [e["accion"] for e in reg]
    assert [p["fuente"] for p in cap["pasos"]] == [e["fuente"] for e in reg]


def test_capturar_oraculo_encadena():
    from atisbo.control import PredictorOraculo
    from atisbo.env.mundo import aplicar
    from atisbo.estado import Accion
    from atisbo.lazo import Mundo
    from atisbo.politica_llm import PoliticaSimulada
    from demo.registro import capturar, estado_desde_dict

    cap = capturar(PredictorOraculo(), PoliticaSimulada(seed=1), Mundo(seed=1, p_perturb=0.0), tau=0.5, epsilon=1e9)
    assert len(cap["pasos"]) > 0
    assert any(p["fuente"] == "atisbo" for p in cap["pasos"])
    assert all(len(p["predichas"]) > 0 for p in cap["pasos"])
    for i in range(len(cap["pasos"]) - 1):
        s = estado_desde_dict(cap["pasos"][i]["estado"])
        s2 = estado_desde_dict(cap["pasos"][i + 1]["estado"])
        esperado = aplicar(s, Accion[cap["pasos"][i]["accion"]])
        assert s2 == esperado


def test_estado_roundtrip_json():
    import json as _json

    from atisbo.env.render import render
    from demo.registro import estado_a_dict, estado_desde_dict

    s = _estado_base()
    d = estado_a_dict(s)
    s2 = estado_desde_dict(_json.loads(_json.dumps(d)))
    assert s2 == s
    assert np.array_equal(render(s), render(s2))


def test_capturar_predichas_restantes():
    from atisbo.control import PredictorOraculo
    from atisbo.lazo import Mundo
    from atisbo.politica_llm import PoliticaSimulada
    from demo.registro import capturar

    cap = capturar(PredictorOraculo(), PoliticaSimulada(seed=1), Mundo(seed=1, p_perturb=0.0), tau=0.5, epsilon=1e9)
    atisbo_pasos = [p for p in cap["pasos"] if p["fuente"] == "atisbo"]
    assert atisbo_pasos
    for p in atisbo_pasos:
        assert p["predichas"]
        assert p["predichas"][0] == p["accion"]
    assert any(len(p["predichas"]) == 1 for p in atisbo_pasos)


def test_figuras_media_fuera_ic_no_falla(tmp_path):
    from demo.figuras import main as fig_main

    resd = tmp_path / "results"
    outd = tmp_path / "out"
    resd.mkdir()
    _resultados_minimos(resd, con_e5=True, con_sam=False)
    e3p = resd / "E3.json"
    e3 = json.loads(e3p.read_text(encoding="utf-8"))
    e3["metricas"]["llamadas_llm@0.05"]["media"] = 100.0
    e3["metricas"]["riesgo_trazas@0.05"]["media"] = 50.0
    e3p.write_text(json.dumps(e3), encoding="utf-8")
    e5p = resd / "E5.json"
    e5 = json.loads(e5p.read_text(encoding="utf-8"))
    e5["celdas"][0]["metricas"]["llamadas_llm"]["media"] = 50.0
    e5["celdas"][0]["metricas"]["exito"]["media"] = 10.0
    e5p.write_text(json.dumps(e5), encoding="utf-8")
    fig_main(["--results", str(resd), "--out", str(outd)])
    for nombre in ["llamadas_vs_alpha.png", "riesgo_vs_alpha.png", "e1_vs_e2.png", "matriz_e5.png"]:
        p = outd / nombre
        assert p.exists() and p.stat().st_size > 1024
