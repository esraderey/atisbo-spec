"""Remates O7 parte 1: lo que se pinta y se guarda. Sin red, con tmp_path."""
from __future__ import annotations

import json

import imageio.v2 as imageio
import matplotlib.axes
import numpy as np
import pytest
from PIL import ImageDraw

import atisbo.exp.comun as comun_mod
import atisbo.lazo as lazo_mod
import atisbo.meta_api as meta_mod
import atisbo.modelo as modelo_mod
import atisbo.politica_llm as pol_mod
import atisbo.sorpresa as sorp_mod
import demo.registro as reg_mod
from atisbo.control import Prediccion, PredictorNulo, PredictorOraculo, PredictorRuido
from atisbo.env.mundo import aplicar, es_exito
from atisbo.env.render import CELL, render
from atisbo.env.trazas import CUBOS_FIJOS
from atisbo.estado import Accion, EstadoSim
from atisbo.exp.comun import split_episodios
from atisbo.lazo import Mundo, correr_episodio
from atisbo.politica_llm import PoliticaSimulada
from demo.figuras import _err, _fig_e1_vs_e2, _fig_llamadas_vs_alpha, _fig_matriz_e5, _fig_riesgo_vs_alpha
from demo.registro import capturar, estado_a_dict, estado_desde_dict
from demo.video_lazo import MARCADOR, _FUENTE, _celdas_fantasma, main as video_main, pintar_paso


class _Fijo:
    """Stub común: acciones fijas, estados lejos, confianza 0.8."""

    def predecir(self, ventana) -> Prediccion:
        d = int(np.asarray(ventana).shape[1])
        return Prediccion(
            acciones=(Accion.DER, Accion.ABAJO, Accion.IZQ),
            estados=np.full((3, d), 1e3, dtype=np.float32),
            confianza=0.8,
        )


def _base(bot=(6, 6), bloqueo=None, paso=0):
    return EstadoSim(
        bot=tuple(bot),
        carga=None,
        objetos=((1, "rojo", 5, 5, False), (2, "verde", 7, 7, False)),
        cubos=dict(CUBOS_FIJOS),
        bloqueo=bloqueo,
        paso=int(paso),
    )


def _paso_fantasma(bot, predichas, bloqueo=None):
    return {"estado": estado_a_dict(_base(bot=bot, bloqueo=bloqueo)), "predichas": list(predichas)}


def _registro_video(n=12, bot=(6, 6)):
    pasos = []
    for i in range(n):
        s = _base(bot=bot, paso=i)
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


def _met(media, lo, hi):
    return {"media": float(media), "ic95": [float(lo), float(hi)], "n": 10}


def _espias_mpl(monkeypatch):
    orig_bar = matplotlib.axes.Axes.bar
    orig_err = matplotlib.axes.Axes.errorbar
    orig_hline = matplotlib.axes.Axes.axhline
    orig_plot = matplotlib.axes.Axes.plot
    bars, errs, hlines, plots = [], [], [], []

    def spy_bar(self, *a, **k):
        bars.append((list(a[0]), list(a[1]), dict(k)))
        return orig_bar(self, *a, **k)

    def spy_err(self, *a, **k):
        errs.append((list(a), dict(k)))
        return orig_err(self, *a, **k)

    def spy_hline(self, *a, **k):
        hlines.append((list(a), dict(k)))
        return orig_hline(self, *a, **k)

    def spy_plot(self, *a, **k):
        plots.append((list(a), dict(k)))
        return orig_plot(self, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy_bar)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", spy_err)
    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", spy_hline)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)
    return bars, errs, hlines, plots


def test_t1_fijo_ciclico_y_sorpresa():
    cap = capturar(_Fijo(), PoliticaSimulada(seed=0), Mundo(seed=1, p_perturb=0.0), tau=0.5, epsilon=1e9)
    assert len(cap["pasos"]) >= 6
    assert all(p["fuente"] == "atisbo" for p in cap["pasos"])
    ciclo = [["DER", "ABAJO", "IZQ"], ["ABAJO", "IZQ"], ["IZQ"]]
    for i, p in enumerate(cap["pasos"]):
        assert list(p["predichas"]) == ciclo[i % 3]

    cap2 = capturar(_Fijo(), PoliticaSimulada(seed=0), Mundo(seed=1, p_perturb=0.0), tau=0.5, epsilon=0.0)
    assert len(cap2["pasos"]) >= 6
    for i, p in enumerate(cap2["pasos"]):
        if i % 2 == 0:
            assert p["fuente"] == "atisbo"
            assert list(p["predichas"]) == ["DER", "ABAJO", "IZQ"]
        else:
            assert p["fuente"] == "llm"
            assert float(p["sorpresa"]) > 0.0
            assert list(p["predichas"]) == ["ABAJO", "IZQ"]


def test_t2_fijo_igual_plain():
    m1 = Mundo(seed=5, p_perturb=0.0)
    m2 = Mundo(seed=5, p_perturb=0.0)
    reg: list = []
    res = correr_episodio(_Fijo(), PoliticaSimulada(seed=0), m1, tau=0.5, epsilon=0.0, registro=reg)
    cap = capturar(_Fijo(), PoliticaSimulada(seed=0), m2, tau=0.5, epsilon=0.0)
    assert len(cap["pasos"]) == len(reg) > 0
    for c, e in zip(cap["pasos"], reg):
        assert c["fuente"] == e["fuente"]
        assert c["accion"] == e["accion"]
        assert float(c["conf"]) == float(e["conf"])
        assert float(c["sorpresa"]) == float(e["sorpresa"])
    assert bool(cap["exito"]) == bool(res["exito"])
    assert int(cap["llamadas_llm"]) == int(res["llamadas_llm"])
    assert any(float(p["conf"]) == 0.8 for p in cap["pasos"])
    assert any(float(p["sorpresa"]) != 0.0 for p in cap["pasos"])


def test_t3_oraculo_horizonte1():
    m1 = Mundo(seed=1, p_perturb=0.0)
    m2 = Mundo(seed=1, p_perturb=0.0)
    reg: list = []
    res = correr_episodio(
        PredictorOraculo(), PoliticaSimulada(seed=1), m1, tau=0.5, epsilon=1e9, registro=reg, horizonte=1
    )
    cap = capturar(
        PredictorOraculo(), PoliticaSimulada(seed=1), m2, tau=0.5, epsilon=1e9, horizonte=1
    )
    assert len(cap["pasos"]) == len(reg) > 0
    assert bool(cap["exito"]) == bool(res["exito"])
    for c, e in zip(cap["pasos"], reg):
        assert c["fuente"] == e["fuente"]
        assert c["accion"] == e["accion"]
        assert float(c["conf"]) == float(e["conf"])
        assert float(c["sorpresa"]) == float(e["sorpresa"])
    assert all(len(p["predichas"]) == 3 for p in cap["pasos"])


def test_t4_transicion_y_exito():
    for pred in (PredictorNulo(), PredictorOraculo()):
        cap = capturar(pred, PoliticaSimulada(seed=1), Mundo(seed=1, p_perturb=0.0), tau=0.5, epsilon=1e9)
        assert len(cap["pasos"]) > 1
        for i in range(len(cap["pasos"]) - 1):
            s = estado_desde_dict(cap["pasos"][i]["estado"])
            s2 = estado_desde_dict(cap["pasos"][i + 1]["estado"])
            assert s2 == aplicar(s, Accion[cap["pasos"][i]["accion"]])
        ultimo = estado_desde_dict(cap["pasos"][-1]["estado"])
        final = estado_desde_dict(cap["estado_final"])
        assert final == aplicar(ultimo, Accion[cap["pasos"][-1]["accion"]])
        assert bool(es_exito(final)) == bool(cap["exito"])


def test_t5_estado_json():
    s = EstadoSim(
        bot=(6, 6),
        carga="azul",
        objetos=((1, "azul", 0, 11, True),),
        cubos=dict(CUBOS_FIJOS),
        bloqueo=(7, 8),
        paso=9,
    )
    d = estado_a_dict(s)
    assert d["bloqueo"] == [7, 8]
    s2 = estado_desde_dict(json.loads(json.dumps(d)))
    assert s2 == s
    assert np.array_equal(render(s), render(s2))


def test_t6_registro_main_espias(tmp_path, monkeypatch):
    gen_ll: list = []
    gen_rets: list = []
    orig_gen = comun_mod.generar_trazas

    def spy_gen(seed, n, p):
        gen_ll.append((int(seed), int(n), float(p)))
        r = orig_gen(int(seed), 4, float(p))
        gen_rets.append(r)
        return r

    monkeypatch.setattr(comun_mod, "generar_trazas", spy_gen)

    ent_ll: list = []
    ent_ret: list = []

    def spy_entrenar(episodios, seed=None, epocas=None, **kw):
        ent_ll.append((episodios, int(seed), int(epocas)))
        m = PredictorRuido(0.3, seed=int(seed))
        ent_ret.append(m)
        return m

    monkeypatch.setattr(modelo_mod.Atisbo, "entrenar", spy_entrenar)

    cal_ll: list = []

    def spy_cal(modelo, eps):
        cal_ll.append((modelo, eps))
        return 1.5

    monkeypatch.setattr(sorp_mod, "calibrar_epsilon", spy_cal)

    mundo_ll: list = []
    OrigMundo = lazo_mod.Mundo

    class SpyMundo(OrigMundo):
        def __init__(self, seed=None, p_perturb=0.0, **kw):
            mundo_ll.append((int(seed), float(p_perturb)))
            super().__init__(seed=int(seed), p_perturb=float(p_perturb))

    monkeypatch.setattr(lazo_mod, "Mundo", SpyMundo)

    cli_ll: list = []

    class SpyCliente:
        def __init__(self, modo=None, cache_dir=None, **kw):
            cli_ll.append((str(modo), str(cache_dir)))

    monkeypatch.setattr(meta_mod, "ClienteMeta", SpyCliente)
    monkeypatch.setattr(pol_mod, "PoliticaLLM", lambda cliente=None, **kw: PoliticaSimulada(seed=0))

    cap_ll: list = []
    cap_rets: list = []
    orig_cap = reg_mod.capturar

    def spy_cap(predictor, politica, mundo, tau=None, epsilon=None, **kw):
        r = orig_cap(predictor, politica, mundo, tau=float(tau), epsilon=float(epsilon), **kw)
        cap_ll.append((predictor, politica, mundo, float(tau), float(epsilon)))
        cap_rets.append(r)
        return r

    monkeypatch.setattr(reg_mod, "capturar", spy_cap)

    out = tmp_path / "reg.json"
    reg_mod.main(["--seed", "3", "--episodio", "2", "--out", str(out)])
    assert gen_ll == [(3, 200, 0.02), (3, 200, 0.0)]
    assert len(ent_ll) == 1
    exp_tr, _ = split_episodios(gen_rets[0], 3)
    assert list(ent_ll[0][0]) == list(exp_tr)
    assert ent_ll[0][1] == 3
    assert ent_ll[0][2] == 10
    _, exp_te = split_episodios(gen_rets[1], 3)
    assert len(cal_ll) == 1
    assert cal_ll[0][0] is ent_ret[0]
    assert list(cal_ll[0][1]) == list(exp_te)
    assert mundo_ll == [(1003, 0.02)]
    assert cli_ll == [("replay", "cache/llm")]
    assert len(cap_ll) == 3
    for _pr, _po, _mu, _tau, _eps in cap_ll:
        assert float(_tau) == 0.5
        assert float(_eps) == 1.5
    assert cap_ll[0][2] is cap_ll[1][2] is cap_ll[2][2]
    assert cap_ll[0][0] is cap_ll[1][0] is cap_ll[2][0]
    assert cap_ll[0][1] is cap_ll[1][1] is cap_ll[2][1]
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert datos["seed"] == 3
    assert datos["episodio"] == 2
    assert float(datos["tau"]) == 0.5
    assert float(datos["epsilon"]) == 1.5
    assert datos["pasos"] == cap_rets[2]["pasos"]
    assert datos["exito"] == cap_rets[2]["exito"]
    assert datos["llamadas_llm"] == cap_rets[2]["llamadas_llm"]
    assert datos["estado_final"] == cap_rets[2]["estado_final"]
    assert datos["pasos"] != cap_rets[0]["pasos"]
    with pytest.raises(ValueError):
        reg_mod.main(["--seed", "3", "--episodio", "-1", "--out", str(tmp_path / "x.json")])


def test_t7_celdas_fantasma():
    assert _celdas_fantasma(_paso_fantasma((6, 6), ["DER", "DER", "ABAJO"])) == [(7, 6), (8, 6), (8, 7)]
    assert _celdas_fantasma(_paso_fantasma((6, 6), ["IZQ", "ARRIBA"])) == [(5, 6), (5, 5)]
    assert _celdas_fantasma(_paso_fantasma((6, 6), ["TOMAR", "SOLTAR", "ESPERAR"])) == [(6, 6), (6, 6), (6, 6)]
    assert _celdas_fantasma(_paso_fantasma((11, 0), ["DER", "ARRIBA"])) == [(11, 0), (11, 0)]
    assert _celdas_fantasma(_paso_fantasma((0, 11), ["IZQ", "ABAJO"])) == [(0, 11), (0, 11)]
    assert _celdas_fantasma(_paso_fantasma((5, 4), ["DER", "DER"], bloqueo=(6, 4))) == [(5, 4), (5, 4)]


def test_t8_fantasma_local():
    s = estado_a_dict(_base())
    base = {"estado": s, "fuente": "atisbo", "accion": "DER", "conf": 0.9, "sorpresa": 0.0, "predichas": []}
    ghost = {"estado": s, "fuente": "atisbo", "accion": "DER", "conf": 0.9, "sorpresa": 0.0, "predichas": ["IZQ", "ARRIBA"]}
    a = pintar_paso(base, 0, 1).astype(int)
    b = pintar_paso(ghost, 0, 1).astype(int)
    ma = a[:384, :, :]
    mb = b[:384, :, :]
    celdas = {(5, 6), (5, 5)}
    for (cx, cy) in celdas:
        px = cx * 32 + 16
        py = cy * 32 + 16
        assert np.any(ma[py - 2:py + 3, px - 2:px + 3, :] != mb[py - 2:py + 3, px - 2:px + 3, :])
    for y in range(384):
        for x in range(384):
            if (x // 32, y // 32) not in celdas:
                assert (ma[y, x] == mb[y, x]).all()


def test_t9_texto_contador(monkeypatch):
    vistos: list = []
    orig = ImageDraw.ImageDraw.text

    def espia(self, xy, texto, *a, **k):
        vistos.append((str(texto), k.get("font")))
        return orig(self, xy, texto, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", espia)
    s = estado_a_dict(_base())
    for fuente in ("atisbo", "llm"):
        vistos.clear()
        pintar_paso({"estado": s, "fuente": fuente, "accion": "DER", "conf": 0.9, "sorpresa": 0.0, "predichas": []}, 3, 7)
        textos = [t for t, _f in vistos]
        assert "llamadas LLM: 3 / pasos: 7" in textos
        assert fuente in textos
        for t, f in vistos:
            if t == "llamadas LLM: 3 / pasos: 7":
                assert f is _FUENTE


def test_t10_fps(tmp_path):
    reg = {"seed": 0, "episodio": 0, "tau": 0.5, "epsilon": 1.0, "pasos": _registro_video(4)}
    jp = tmp_path / "reg.json"
    jp.write_text(json.dumps(reg), encoding="utf-8")
    o1 = tmp_path / "a.mp4"
    video_main(["--episodio", str(jp), "--out", str(o1)])
    with imageio.get_reader(str(o1)) as lector:
        assert float(lector.get_meta_data()["fps"]) == pytest.approx(4.0)
    o2 = tmp_path / "b.mp4"
    video_main(["--episodio", str(jp), "--out", str(o2), "--fps", "2"])
    with imageio.get_reader(str(o2)) as lector:
        assert float(lector.get_meta_data()["fps"]) == pytest.approx(2.0)


def test_t11_estado_final_frame(tmp_path):
    pasos = _registro_video(12)
    reg = {"seed": 0, "episodio": 0, "tau": 0.5, "epsilon": 1.0, "pasos": pasos,
           "estado_final": estado_a_dict(_base(bot=(1, 2), paso=12))}
    jp = tmp_path / "reg.json"
    jp.write_text(json.dumps(reg), encoding="utf-8")
    out = tmp_path / "lazo.mp4"
    video_main(["--episodio", str(jp), "--out", str(out)])
    with imageio.get_reader(str(out)) as lector:
        frames = [np.asarray(f) for f in lector]
    assert len(frames) == 13
    ultimo = np.asarray(frames[-1]).astype(int)
    px = ultimo[2 * CELL + CELL // 2, 1 * CELL + CELL // 2]
    assert abs(int(px[0]) - 64) <= 12 and abs(int(px[1]) - 64) <= 12 and abs(int(px[2]) - 64) <= 12
    x0, y0, x1, y1 = MARCADOR
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    mk = ultimo[cy, cx].astype(int)
    assert pasos[-1]["fuente"] == "llm"
    assert mk[0] > mk[1] and mk[0] > mk[2]

    reg2 = {"seed": 0, "episodio": 0, "tau": 0.5, "epsilon": 1.0, "pasos": pasos}
    jp2 = tmp_path / "reg2.json"
    jp2.write_text(json.dumps(reg2), encoding="utf-8")
    out2 = tmp_path / "lazo2.mp4"
    video_main(["--episodio", str(jp2), "--out", str(out2)])
    with imageio.get_reader(str(out2)) as lector:
        frames2 = [np.asarray(f) for f in lector]
    assert len(frames2) == 12


def test_t12_llamadas_vs_alpha(tmp_path, monkeypatch):
    bars, _errs, hlines, _plots = _espias_mpl(monkeypatch)
    e1 = {"metricas": {"llamadas_llm": _met(11.0, 10.0, 12.5)}}
    e2 = {"metricas": {"llamadas_llm": _met(3.25, 2.75, 3.9)}}
    alphas = [0.10, 0.01, 0.05]
    medias = [2.5, 5.5, 3.75]
    los = [2.0, 5.0, 3.0]
    his = [3.1, 6.2, 4.1]
    metricas = {f"llamadas_llm@{float(a)}": _met(m, lo, hi) for a, m, lo, hi in zip(alphas, medias, los, his)}
    e3 = {"por_alpha": [{"alpha": float(a)} for a in alphas], "metricas": metricas}
    _fig_llamadas_vs_alpha(e1, e2, e3, tmp_path / "ll.png")
    assert len(bars) == 1
    xs, alturas, kw = bars[0]
    assert xs == ["0.01", "0.05", "0.1"]
    assert alturas == pytest.approx([5.5, 3.75, 2.5])
    yerr = kw["yerr"]
    assert np.allclose(np.asarray(yerr, dtype=float), [[0.5, 0.75, 0.5], [0.7, 0.35, 0.6]])
    assert len(hlines) == 2
    assert {h[1]["label"]: h[0][0] for h in hlines} == {"E1": 11.0, "E2": 3.25}


def test_t12_riesgo_vs_alpha(tmp_path, monkeypatch):
    _bars, errs, _hlines, plots = _espias_mpl(monkeypatch)
    alphas = [0.10, 0.01, 0.05]
    rt = [0.07, 0.005, 0.03]
    rt_lo = [0.06, 0.003, 0.028]
    rt_hi = [0.08, 0.007, 0.033]
    re = [0.075, 0.006, 0.028]
    re_lo = [0.070, 0.004, 0.025]
    re_hi = [0.080, 0.009, 0.030]
    metricas = {}
    for a, m, lo, hi in zip(alphas, rt, rt_lo, rt_hi):
        metricas[f"riesgo_trazas@{float(a)}"] = _met(m, lo, hi)
    for a, m, lo, hi in zip(alphas, re, re_lo, re_hi):
        metricas[f"riesgo_empirico@{float(a)}"] = _met(m, lo, hi)
    e3 = {"por_alpha": [{"alpha": float(a)} for a in alphas], "metricas": metricas}
    _fig_riesgo_vs_alpha(e3, tmp_path / "ri.png")
    assert len(errs) == 2
    por_etiqueta = {str(e[1].get("label")): e for e in errs}
    assert set(por_etiqueta) == {"riesgo_trazas", "riesgo_empirico"}
    e_rt = por_etiqueta["riesgo_trazas"]
    assert np.allclose(np.asarray(e_rt[0][1], dtype=float), [0.005, 0.03, 0.07])
    assert np.allclose(np.asarray(e_rt[1]["yerr"], dtype=float), [[0.002, 0.002, 0.01], [0.002, 0.003, 0.01]])
    e_re = por_etiqueta["riesgo_empirico"]
    assert np.allclose(np.asarray(e_re[0][1], dtype=float), [0.006, 0.028, 0.075])
    assert np.allclose(np.asarray(e_re[1]["yerr"], dtype=float), [[0.002, 0.003, 0.005], [0.003, 0.002, 0.005]])
    assert len(plots) == 1
    xp, yp = plots[0][0][0], plots[0][0][1]
    assert np.allclose(np.asarray(xp, dtype=float), [0.01, 0.05, 0.10])
    assert np.allclose(np.asarray(yp, dtype=float), [0.01, 0.05, 0.10])


def test_t12_e1_vs_e2(tmp_path, monkeypatch):
    e1 = {"metricas": {"llamadas_llm": _met(11.0, 10.0, 12.5), "exito": _met(0.81, 0.75, 0.86)}}
    e2 = {"metricas": {"llamadas_llm": _met(3.25, 2.75, 3.9), "exito": _met(0.77, 0.70, 0.83)}}
    e2sam = {"seeds": [0, 1], "config": {"seeds_evaluadas": [0, 1]},
             "metricas": {"llamadas_llm": _met(4.5, 4.0, 5.2), "exito": _met(0.71, 0.65, 0.78)}}
    bars, _errs, _hlines, _plots = _espias_mpl(monkeypatch)
    _fig_e1_vs_e2(e1, e2, None, tmp_path / "a.png")
    assert len(bars) == 2
    assert bars[0][0] == ["E1", "E2"]
    assert bars[0][1] == pytest.approx([11.0, 3.25])
    assert bars[1][0] == ["E1", "E2"]
    assert bars[1][1] == pytest.approx([0.81, 0.77])
    bars.clear()
    _fig_e1_vs_e2(e1, e2, e2sam, tmp_path / "b.png")
    assert len(bars) == 2
    assert bars[0][0] == ["E1", "E2", "E2_sam (semillas 0, 1)"]
    assert bars[0][1] == pytest.approx([11.0, 3.25, 4.5])
    assert bars[1][0] == ["E1", "E2", "E2_sam (semillas 0, 1)"]
    assert bars[1][1] == pytest.approx([0.81, 0.77, 0.71])


def test_t12_matriz_e5(tmp_path, monkeypatch):
    bars, _errs, _hlines, _plots = _espias_mpl(monkeypatch)
    c0 = {"nombre": "p0", "config": {"p": 0.0},
          "metricas": {"llamadas_llm": _met(2.5, 2.0, 3.1), "exito": _met(0.85, 0.80, 0.89)}}
    c2 = {"nombre": "p002", "config": {"p": 0.02},
          "metricas": {"llamadas_llm": _met(3.75, 3.0, 4.1), "exito": _met(0.77, 0.70, 0.83)}}
    c5 = {"nombre": "p005", "config": {"p": 0.05}, "parcial": "solo episodios en caché",
          "metricas": {"llamadas_llm": _met(4.5, 4.0, 5.2), "exito": _met(0.65, 0.60, 0.71)}}
    omit = {"nombre": "p010", "config": {"p": 0.10}, "metricas": {}}
    e5 = {"celdas": [c5, omit, c0, c2]}
    _fig_matriz_e5(e5, tmp_path / "m.png")
    assert len(bars) == 2
    assert bars[0][0] == ["p=0.0", "p=0.02", "p=0.05 (parcial)"]
    assert bars[0][1] == pytest.approx([2.5, 3.75, 4.5])
    assert bars[1][0] == ["p=0.0", "p=0.02", "p=0.05 (parcial)"]
    assert bars[1][1] == pytest.approx([0.85, 0.77, 0.65])


def test_t12_err_recorta():
    assert _err([5.0, 10.0], [6.0, 9.0], [6.5, 9.5]) == [[0.0, 1.0], [1.5, 0.0]]
