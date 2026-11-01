"""Remates O5 (revisor ciego + mutaciones). Sin red. Primero en rojo."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from atisbo.env.render import BLOQUEO_COLOR, BOT_COLOR, CELL, COLORES, render
from atisbo.env.rutina import elegir_accion
from atisbo.env.trazas import CUBOS_FIJOS
from atisbo.control import PredictorNulo
from atisbo.estado import Accion, EstadoSim, estado_desde_mascaras
from atisbo.exp import e2
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import MetaAPIError
from atisbo.politica_llm import PoliticaSimulada
from atisbo.sam import ClienteSam, MundoSam, SeguidorEstado


def _mascaras_desde_frame(frame) -> dict[int, np.ndarray]:
    arr = np.asarray(frame, dtype=np.uint8)
    out: dict[int, np.ndarray] = {}
    oid = 1
    for cy in range(12):
        for cx in range(12):
            y0, y1 = cy * CELL, (cy + 1) * CELL
            x0, x1 = cx * CELL, (cx + 1) * CELL
            celda = arr[y0:y1, x0:x1]
            m_bot = np.all(celda == np.asarray(BOT_COLOR, dtype=np.uint8), axis=2)
            if m_bot.any():
                full = np.zeros((384, 384), dtype=bool)
                full[y0:y1, x0:x1] = m_bot
                out[oid] = full
                oid += 1
                continue
            zona = celda[8:25, 8:25]
            for _tipo, col in COLORES.items():
                m = np.all(zona == np.asarray(col, dtype=np.uint8), axis=2)
                if m.any():
                    full = np.zeros((384, 384), dtype=bool)
                    full[y0 + 8:y0 + 25, x0 + 8:x0 + 25] = m
                    out[oid] = full
                    oid += 1
            m_bl = np.all(celda == np.asarray(BLOQUEO_COLOR, dtype=np.uint8), axis=2)
            if m_bl.any():
                full = np.zeros((384, 384), dtype=bool)
                full[y0:y1, x0:x1] = m_bl
                out[oid] = full
                oid += 1
    return out


def _solo_bot(frame) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for oid, m in _mascaras_desde_frame(frame).items():
        ys, xs = np.nonzero(np.asarray(m, dtype=bool))
        if len(ys) == 0:
            continue
        lado = max(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)
        if lado >= 22:
            out[oid] = m
    return out


def _mascaras_sin_bloqueo(frame) -> dict[int, np.ndarray]:
    arr = np.asarray(frame, dtype=np.uint8)
    out: dict[int, np.ndarray] = {}
    oid = 1
    for cy in range(12):
        for cx in range(12):
            y0, y1 = cy * CELL, (cy + 1) * CELL
            x0, x1 = cx * CELL, (cx + 1) * CELL
            celda = arr[y0:y1, x0:x1]
            m_bot = np.all(celda == np.asarray(BOT_COLOR, dtype=np.uint8), axis=2)
            if m_bot.any():
                full = np.zeros((384, 384), dtype=bool)
                full[y0:y1, x0:x1] = m_bot
                out[oid] = full
                oid += 1
                continue
            zona = celda[8:25, 8:25]
            for _tipo, col in COLORES.items():
                m = np.all(zona == np.asarray(col, dtype=np.uint8), axis=2)
                if m.any():
                    full = np.zeros((384, 384), dtype=bool)
                    full[y0 + 8:y0 + 25, x0 + 8:x0 + 25] = m
                    out[oid] = full
                    oid += 1
    return out


def _sse_cuerpo(chunks, uso=None, completado=True, tipo_final="response.completed"):
    lineas = ["data: " + json.dumps({"type": "response.output_text.delta", "delta": c}) for c in chunks]
    if completado:
        if tipo_final == "response.completed":
            lineas.append("data: " + json.dumps({"type": "response.completed", "response": {"usage": uso or {}}}))
        else:
            lineas.append("data: " + json.dumps({"type": tipo_final}))
    lineas.append("data: [DONE]")
    return ("\n".join(lineas) + "\n").encode("utf-8")


class _Fake:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append(url)
        idx = min(len(self.llamadas) - 1, len(self.respuestas) - 1)
        r = self.respuestas[idx]
        if isinstance(r, BaseException):
            raise r
        return r


class _Mixto:
    def __init__(self):
        self.llamadas = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append(url)
        if "chat/completions" in url:
            body = json.dumps({"choices": [{"message": {"content": '{"accion":"ESPERAR"}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode("utf-8")
            return (200, {}, body)
        return (200, {}, _sse_cuerpo([], {"images_processed": 1}))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _texto_real() -> str:
    fixdir = Path("tests/fixtures/sam")
    for f in sorted(fixdir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("payload", {}).get("medio") == "imagen":
            return str(d["texto"])
    raise AssertionError("sin fixture imagen")


def test_r1_soltar_sin_bot():
    seg = SeguidorEstado()
    s0 = EstadoSim(bot=(0, 0), carga="rojo", objetos=(), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    seg.actualizar(s0)
    visto = EstadoSim(bot=None, carga=None, objetos=(), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    r1 = seg.actualizar(visto, Accion.SOLTAR)
    assert r1.carga is None
    assert len(r1.objetos) == 1
    _id, tipo, x, y, en_cubo = r1.objetos[0]
    assert tipo == "rojo"
    assert (x, y) == (0, 0)
    assert bool(en_cubo) is True
    last = {"a": None}

    def perc(frame):
        if last["a"] is Accion.SOLTAR:
            return {}
        return _mascaras_desde_frame(frame)

    ms = MundoSam(Mundo(seed=1000, p_perturb=0.0), perc)
    s = ms.reiniciar()
    for _ in range(160):
        if ms.exito:
            break
        a = elegir_accion(s)
        last["a"] = a
        s = ms.avanzar(s, a)
    assert ms.exito is True


def test_r2_tomar_fallido():
    s0 = EstadoSim(bot=(5, 5), carga=None, objetos=((1, "rojo", 5, 5, False),), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    seg = SeguidorEstado()
    seg.actualizar(s0)
    visto = EstadoSim(bot=(5, 5), carga=None, objetos=(), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    r_fail = seg.actualizar(visto, Accion.TOMAR)
    assert tuple(r_fail.objetos) == ()
    seg2 = SeguidorEstado()
    seg2.actualizar(s0)
    r_tap = seg2.actualizar(visto, Accion.ESPERAR)
    assert len(r_tap.objetos) == 1


def test_r3_falta_un_frame():
    s0 = EstadoSim(bot=(0, 0), carga=None, objetos=((1, "rojo", 2, 2, False),), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    seg = SeguidorEstado()
    seg.actualizar(s0)
    vacio = EstadoSim(bot=(0, 0), carga=None, objetos=(), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    r1 = seg.actualizar(vacio, Accion.ESPERAR)
    assert len(r1.objetos) == 1
    assert r1.objetos[0][0] == 1
    r2 = seg.actualizar(vacio, Accion.ESPERAR)
    assert tuple(r2.objetos) == ()
    seg3 = SeguidorEstado()
    seg3.actualizar(s0)
    seg3.actualizar(vacio, Accion.ESPERAR)
    r3 = seg3.actualizar(s0, Accion.ESPERAR)
    assert len(r3.objetos) == 1
    assert r3.objetos[0][0] == 1
    calls = {"n": 0}

    def perc(frame):
        n = calls["n"]
        calls["n"] += 1
        if n == 1:
            return _solo_bot(frame)
        return _mascaras_desde_frame(frame)

    ms = MundoSam(Mundo(seed=1000, p_perturb=0.0), perc)
    a0 = ms.reiniciar()
    n0 = len(a0.objetos)
    assert n0 > 0
    a1 = ms.avanzar(a0, Accion.ESPERAR)
    assert len(a1.objetos) == n0


def test_r4_rutinas_y_malas_real(tmp_path, monkeypatch):
    ms = MundoSam(Mundo(seed=1000, p_perturb=0.0), _mascaras_desde_frame)
    s = ms.reiniciar()
    assert list(ms.rutinas_reales) == []
    s = ms.avanzar(s, Accion.ESPERAR)
    assert len(list(ms.rutinas_reales)) == 1
    ms.reiniciar()
    assert list(ms.rutinas_reales) == []
    import atisbo.exp.e2 as e2mod

    real = e2mod.correr_episodio
    regs: list = []
    ruts: list = []

    def spy(predictor, politica, mundo, tau, epsilon, registro=None):
        if registro is None:
            registro = []
        n0 = len(registro)
        res = real(predictor, politica, mundo, tau=tau, epsilon=epsilon, registro=registro)
        regs.append(list(registro[n0:]))
        ruts.append(list(mundo.rutinas_reales))
        return res

    monkeypatch.setattr(e2mod, "correr_episodio", spy)
    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "r4.json"
    e2.main(["--rapido", "--tau", "0.0", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=_solo_bot)
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert "malas_aceptadas_real" in datos["metricas"]
    assert "tasa_malas_aceptadas_real" in datos["metricas"]
    assert len(regs) == len(ruts) > 0
    malas_list: list[float] = []
    tasa_list: list[float] = []
    for reg, rt in zip(regs, ruts):
        assert len(reg) == len(rt)
        m = sum(1 for ee, rr in zip(reg, rt, strict=True) if ee.get("fuente") == "atisbo" and ee.get("accion") != rr)
        malas_list.append(float(m))
        tasa_list.append(float(m / len(reg)) if reg else 0.0)
    exp_m = sum(malas_list) / len(malas_list)
    exp_t = sum(tasa_list) / len(tasa_list)
    assert datos["metricas"]["malas_aceptadas_real"]["media"] == pytest.approx(exp_m)
    assert datos["metricas"]["tasa_malas_aceptadas_real"]["media"] == pytest.approx(exp_t)
    assert datos["metricas"]["malas_aceptadas_real"]["media"] != datos["metricas"]["malas_aceptadas"]["media"]


def test_r5_tope_y_truncado(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "CLAVE-TEST-123")
    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "tope.json"
    e2.main(["--rapido", "--modo", "api", "--estado", "sam", "--max-sam-calls", "5", "--cache", str(tmp_path / "llm1"), "--cache-sam", str(tmp_path / "sam1"), "--out", str(out), "--e1", str(e1_path)], transporte=_Mixto())
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert datos["truncada"] is True
    assert "comparacion_e2" not in datos
    assert any("corrida truncada por tope SAM" in n for n in datos["notas"])
    out2 = tmp_path / "llm1.json"
    e2.main(["--rapido", "--modo", "api", "--estado", "sam", "--max-llm-calls", "1", "--tau", "1.0", "--cache", str(tmp_path / "llm2"), "--cache-sam", str(tmp_path / "sam2"), "--out", str(out2), "--e1", str(e1_path)], transporte=_Mixto())
    d2 = json.loads(out2.read_text(encoding="utf-8"))
    assert d2["truncada"] is True
    assert "comparacion_e2" not in d2


def test_r6_referencia(tmp_path, monkeypatch):
    import atisbo.modelo as mod

    llamadas = {"n": 0}
    real = mod.Atisbo.entrenar

    def spy(*a, **k):
        llamadas["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod.Atisbo, "entrenar", spy)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"config": {"seeds_evaluadas": [0]}, "metricas": {}, "commit": "x", "llm": {"modo": "simulada"}}), encoding="utf-8")
    with pytest.raises(ValueError):
        e2.main(["--rapido", "--seeds", "1", "--modo", "simulada", "--estado", "sam", "--out", str(tmp_path / "o.json"), "--e1", str(_repo_root() / "results" / "E1.json"), "--e2", str(ref)], mascaras_de=_mascaras_desde_frame)
    assert llamadas["n"] == 0
    with pytest.raises(FileNotFoundError):
        e2.main(["--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(tmp_path / "o2.json"), "--e1", str(_repo_root() / "results" / "E1.json"), "--e2", str(tmp_path / "noexiste.json")], mascaras_de=_mascaras_desde_frame)
    assert llamadas["n"] == 0


def _escribe_ref(path: Path, tau: float) -> None:
    data = {"config": {"seeds": [0], "seeds_evaluadas": [0], "n_episodios_por_seed": 1, "p_perturb": 0.02, "tau": float(tau), "n_trazas": 20, "epocas": 2, "rapido": False}, "metricas": {"llamadas_llm": {"por_seed": [3.0]}, "exito": {"por_seed": [1.0]}}, "commit": "abc", "llm": {"modo": "simulada"}}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_r7_comparable(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    ref_ok = tmp_path / "ref_ok.json"
    _escribe_ref(ref_ok, 0.5)
    out = tmp_path / "o.json"
    e2.main(["--seeds", "0", "--n", "1", "--n-trazas", "20", "--epocas", "2", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path), "--e2", str(ref_ok)], mascaras_de=_mascaras_desde_frame)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["comparacion_e2"]["comparable"] is True
    ref_bad = tmp_path / "ref_bad.json"
    _escribe_ref(ref_bad, 0.9)
    out2 = tmp_path / "o2.json"
    e2.main(["--seeds", "0", "--n", "1", "--n-trazas", "20", "--epocas", "2", "--modo", "simulada", "--estado", "sam", "--out", str(out2), "--e1", str(e1_path), "--e2", str(ref_bad)], mascaras_de=_mascaras_desde_frame)
    d2 = json.loads(out2.read_text(encoding="utf-8"))
    assert d2["comparacion_e2"]["comparable"] is False


def test_r8_invalidas(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "CLAVE-TEST-123")
    uso = {"images_processed": 1}
    cuerpos = [_sse_cuerpo(["hola mundo libre"], uso), _sse_cuerpo(["<0f>0<|box;x1=1"], uso), _sse_cuerpo([], uso, tipo_final="response.refusal.done")]
    for i, cuerpo in enumerate(cuerpos):
        d = tmp_path / f"c{i}"
        d.mkdir()
        fake = _Fake([(200, {}, cuerpo)])
        cli = ClienteSam(modo="api", cache_dir=str(d), transporte=fake, dormir=lambda s: None)
        with pytest.raises(MetaAPIError):
            cli.segmentar([np.zeros((8, 8, 3), dtype=np.uint8)])
        assert list(d.glob("*.json")) == []


def test_r9_reintentos(tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.setenv("META_API_KEY", "CLAVE-TEST-123")
    uso = {"images_processed": 1}
    real = _texto_real()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    fake = _Fake([(429, {}, b"a"), (429, {}, b"b"), (429, {}, b"c")])
    cli = ClienteSam(modo="api", cache_dir=str(tmp_path / "a"), transporte=fake, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli.segmentar([frame])
    assert len(fake.llamadas) == 3
    got = {}

    def _dormir(s):
        got["s"] = s

    fake2 = _Fake([(429, {"Retry-After": "7"}, b"x"), (200, {}, _sse_cuerpo([real], uso))])
    cli2 = ClienteSam(modo="api", cache_dir=str(tmp_path / "b"), transporte=fake2, dormir=_dormir)
    assert cli2.segmentar([frame]) == real
    assert got["s"] == 7.0
    fake3 = _Fake([(502, {}, b"bad"), (200, {}, _sse_cuerpo([real], uso))])
    cli3 = ClienteSam(modo="api", cache_dir=str(tmp_path / "c"), transporte=fake3, dormir=lambda s: None)
    assert cli3.segmentar([frame]) == real
    fake4 = _Fake([urllib.error.URLError("caida"), (200, {}, _sse_cuerpo([real], uso))])
    cli4 = ClienteSam(modo="api", cache_dir=str(tmp_path / "d"), transporte=fake4, dormir=lambda s: None)
    assert cli4.segmentar([frame]) == real
    fake5 = _Fake([(200, {}, _sse_cuerpo([], {}, tipo_final="response.failed")), (200, {}, _sse_cuerpo([real], uso))])
    cli5 = ClienteSam(modo="api", cache_dir=str(tmp_path / "e"), transporte=fake5, dormir=lambda s: None)
    assert cli5.segmentar([frame]) == real


def _sha_frames(frames) -> str:
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_r10_latencia():
    mundo = Mundo(seed=1000, p_perturb=0.02)
    s = mundo.reiniciar()
    estados = [s]
    while len(estados) < 60:
        s = mundo.avanzar(s, elegir_accion(s))
        estados.append(s)
    frames = [render(x) for x in estados]
    fixdir = Path("tests/fixtures/sam")
    h_vid = _sha_frames(frames)
    h_img = _sha_frames([frames[0]])
    lat_vid = None
    lat_img = None
    for f in fixdir.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("payload", {}).get("frames") == h_vid:
            lat_vid = float(d["latencia_ms"])
        if d.get("payload", {}).get("frames") == h_img:
            lat_img = float(d["latencia_ms"])
    assert lat_vid is not None
    assert lat_img is not None
    cli = ClienteSam(modo="replay", cache_dir=str(fixdir))
    cli.segmentar(frames)
    cli.segmentar([frames[0]])
    assert cli.latencia_ms_media() == pytest.approx((lat_vid + lat_img) / 2.0)


def test_r11_dos_episodios():
    ms = MundoSam(Mundo(seed=1000, p_perturb=0.05), _mascaras_desde_frame)
    pol = PoliticaSimulada(seed=0)
    r1 = correr_episodio(PredictorNulo(), pol, ms, tau=0.5, epsilon=0.1, registro=[])
    assert ms.frames == r1["pasos"] + 1
    assert ms.pasos_ciegos == 0
    assert ms.coincidencias()["todo"] >= 0.9
    r2 = correr_episodio(PredictorNulo(), pol, ms, tau=0.5, epsilon=0.1, registro=[])
    assert ms.frames == r2["pasos"] + 1
    assert ms.pasos_ciegos == 0
    assert ms.coincidencias()["todo"] >= 0.9
    assert ms.reiniciar().paso == 0


def test_r12_media_dos(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    e2_path = _repo_root() / "results" / "E2.json"
    out = tmp_path / "dos.json"
    e2.main(["--seeds", "0,1", "--n", "1", "--n-trazas", "20", "--epocas", "2", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=_mascaras_desde_frame)
    d = json.loads(out.read_text(encoding="utf-8"))
    ref = json.loads(e2_path.read_text(encoding="utf-8"))
    m0 = float(ref["metricas"]["llamadas_llm"]["por_seed"][0])
    m1 = float(ref["metricas"]["llamadas_llm"]["por_seed"][1])
    assert d["comparacion_e2"]["llamadas_llm_media"] == pytest.approx((m0 + m1) / 2.0)
    assert d["comparacion_e2"]["llamadas_llm_media"] != pytest.approx(m0)
    assert len(d["metricas"]["coincidencia_todo"]["por_seed"]) == 2
    assert all(v is not None for v in d["metricas"]["coincidencia_todo"]["por_seed"])


def test_r13_ciego(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "tau0.json"
    e2.main(["--rapido", "--tau", "0.0", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=lambda frame: {})
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["llamadas_llm"] > 0


def test_r14_semilla(tmp_path, monkeypatch):
    import atisbo.exp.e2 as e2mod
    import atisbo.lazo as lazo_mod

    vistos: list[int] = []
    Real = lazo_mod.Mundo

    def spy(seed=0, p_perturb=0.0):
        vistos.append(int(seed))
        return Real(seed=seed, p_perturb=p_perturb)

    monkeypatch.setattr(lazo_mod, "Mundo", spy)
    monkeypatch.setattr(e2mod, "Mundo", spy)
    e1_path = _repo_root() / "results" / "E1.json"
    e2.main(["--seeds", "3", "--n", "1", "--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(tmp_path / "o.json"), "--e1", str(e1_path)], mascaras_de=_mascaras_desde_frame)
    assert vistos == [1003]


def test_r15_args():
    out_sam = e2._parse_args(["--estado", "sam"])[8]
    assert out_sam == "results/E2_sam.json"
    out_sim = e2._parse_args([])[8]
    assert out_sim == "results/E2.json"


def test_r16_bot_todo(tmp_path):
    ms = MundoSam(Mundo(seed=1000, p_perturb=0.0), _solo_bot)
    s = ms.reiniciar()
    for _ in range(3):
        s = ms.avanzar(s, Accion.ESPERAR)
    c = ms.coincidencias()
    assert c["bot"] == 1.0
    assert c["objetos"] == 0.0
    assert c["todo"] == 0.0

    class _Bloq:
        def __init__(self, m):
            self._m = m

        @property
        def exito(self):
            return self._m.exito

        def reiniciar(self):
            return replace(self._m.reiniciar(), bloqueo=(5, 5))

        def avanzar(self, s, a):
            return replace(self._m.avanzar(s, a), bloqueo=(5, 5))

    ms2 = MundoSam(_Bloq(Mundo(seed=1000, p_perturb=0.0)), _mascaras_sin_bloqueo)
    t = ms2.reiniciar()
    for _ in range(3):
        t = ms2.avanzar(t, Accion.ESPERAR)
    c2 = ms2.coincidencias()
    assert c2["bot"] == 1.0
    assert c2["objetos"] == 1.0
    assert c2["todo"] == 0.0
    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "o.json"
    e2.main(["--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=_solo_bot)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["metricas"]["coincidencia_todo"]["media"] < d["metricas"]["coincidencia_bot"]["media"]


def test_r17_orden_ids():
    s_in = EstadoSim(bot=(5, 5), carga=None, objetos=((99, "rojo", 3, 2, False), (5, "rojo", 1, 1, False)), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    fr = render(s_in)
    visto = estado_desde_mascaras(_mascaras_desde_frame(fr), fr)
    assert [(t, x, y) for _, t, x, y, _ in visto.objetos] == [("rojo", 1, 1), ("rojo", 3, 2)]
    assert [o[0] for o in visto.objetos] == [1, 2]
    seg = SeguidorEstado()
    s0 = EstadoSim(bot=(4, 4), carga=None, objetos=((1, "rojo", 4, 4, False), (2, "rojo", 4, 4, False)), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    seg.actualizar(s0)
    visto2 = EstadoSim(bot=(4, 4), carga="rojo", objetos=(), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    r = seg.actualizar(visto2, Accion.TOMAR)
    assert [o[0] for o in r.objetos] == [2]

# O5 remates m1-m7 (tercera ronda mutaciones)


def test_m1_orden_fila():
    s_in = EstadoSim(bot=(5, 5), carga=None, objetos=((99, "rojo", 3, 1, False), (5, "rojo", 1, 2, False)), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    fr = render(s_in)
    visto = estado_desde_mascaras(_mascaras_desde_frame(fr), fr)
    assert [(t, x, y) for _, t, x, y, _ in visto.objetos] == [("rojo", 3, 1), ("rojo", 1, 2)]
    assert [o[0] for o in visto.objetos] == [1, 2]


def test_m2_encima_conserva_id():
    s0 = EstadoSim(bot=(4, 4), carga=None, objetos=((1, "rojo", 4, 4, False),), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    seg = SeguidorEstado()
    seg.actualizar(s0)
    visto = EstadoSim(bot=(4, 4), carga=None, objetos=(), cubos=dict(CUBOS_FIJOS), bloqueo=None, paso=0)
    r = None
    for _ in range(3):
        r = seg.actualizar(visto, Accion.ESPERAR)
        assert len(r.objetos) == 1
        assert r.objetos[0][0] == 1
        assert (r.objetos[0][2], r.objetos[0][3]) == (4, 4)


def test_m3_rutinas_antes():
    ms = MundoSam(Mundo(1000, 0.0), _mascaras_desde_frame)
    s = ms.reiniciar()
    esperadas: list[str] = []
    for _ in range(20):
        acc = elegir_accion(ms._real)
        esperadas.append(acc.name)
        s = ms.avanzar(s, acc)
    assert list(ms.rutinas_reales) == esperadas


def test_m4_refusal_done_con_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "CLAVE-TEST-123")
    uso = {"images_processed": 1}
    cuerpo = ("data: " + json.dumps({"type": "response.refusal.done"}) + "\n" + "data: " + json.dumps({"type": "response.completed", "response": {"usage": uso}}) + "\n" + "data: [DONE]\n").encode("utf-8")
    d = tmp_path / "m4"
    d.mkdir()
    fake = _Fake([(200, {}, cuerpo)])
    cli = ClienteSam(modo="api", cache_dir=str(d), transporte=fake, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli.segmentar([np.zeros((8, 8, 3), dtype=np.uint8)])
    assert list(d.glob("*.json")) == []


def test_m5_comparable_solo_rapido(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    ref = tmp_path / "ref_m5.json"
    ref.write_text(json.dumps({"config": {"seeds": [0], "seeds_evaluadas": [0], "n_episodios_por_seed": 2, "p_perturb": 0.02, "tau": 0.5, "n_trazas": 20, "epocas": 2, "rapido": False}, "metricas": {"llamadas_llm": {"por_seed": [3.0]}, "exito": {"por_seed": [1.0]}}, "commit": "abc", "llm": {"modo": "simulada"}}), encoding="utf-8")
    out = tmp_path / "o.json"
    e2.main(["--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path), "--e2", str(ref)], mascaras_de=_mascaras_desde_frame)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["comparacion_e2"]["comparable"] is False


def test_m6_malas_solo_atisbo(tmp_path, monkeypatch):
    import atisbo.exp.e2 as e2mod

    real = e2mod.correr_episodio
    regs: list = []
    ruts: list = []

    def spy(predictor, politica, mundo, tau, epsilon, registro=None):
        if registro is None:
            registro = []
        n0 = len(registro)
        res = real(predictor, politica, mundo, tau=tau, epsilon=epsilon, registro=registro)
        regs.append(list(registro[n0:]))
        ruts.append(list(mundo.rutinas_reales))
        return res

    monkeypatch.setattr(e2mod, "correr_episodio", spy)
    calls = {"n": 0}

    def _solo_bot_cuarto(frame):
        n = calls["n"]
        calls["n"] += 1
        if n % 4 == 0:
            return {}
        return _solo_bot(frame)

    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "m6.json"
    e2.main(["--rapido", "--tau", "0.0", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=_solo_bot_cuarto)
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert len(regs) == len(ruts) > 0
    fuentes = {ee.get("fuente") for reg in regs for ee in reg}
    assert "atisbo" in fuentes
    assert "llm" in fuentes
    malas_f: list[float] = []
    malas_sin: list[float] = []
    for reg, rt in zip(regs, ruts, strict=True):
        assert len(reg) == len(rt)
        mf = sum(1 for ee, rr in zip(reg, rt, strict=True) if ee.get("fuente") == "atisbo" and ee.get("accion") != rr)
        ms = sum(1 for ee, rr in zip(reg, rt, strict=True) if ee.get("accion") != rr)
        malas_f.append(float(mf))
        malas_sin.append(float(ms))
    exp_f = sum(malas_f) / len(malas_f)
    exp_s = sum(malas_sin) / len(malas_sin)
    assert datos["metricas"]["malas_aceptadas_real"]["media"] == pytest.approx(exp_f)
    assert exp_f != exp_s
    assert float(datos["metricas"]["malas_aceptadas_real"]["media"]) != float(exp_s)


def _mascaras_todo_en_cubo(frame) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    oid = 1
    for _tipo, (cx, cy) in CUBOS_FIJOS.items():
        full = np.zeros((384, 384), dtype=bool)
        full[cy * CELL:cy * CELL + 4, cx * CELL:cx * CELL + 17] = True
        out[oid] = full
        oid += 1
    return out


def test_m7_falsos_exitos(tmp_path):
    e1_path = _repo_root() / "results" / "E1.json"
    out = tmp_path / "m7.json"
    e2.main(["--rapido", "--modo", "simulada", "--estado", "sam", "--out", str(out), "--e1", str(e1_path)], mascaras_de=_mascaras_todo_en_cubo)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["sam"]["falsos_exitos"] == 2
    assert d["sam"]["exitos_no_detectados"] == 0
