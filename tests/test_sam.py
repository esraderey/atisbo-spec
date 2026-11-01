"""Anclas SAM 3.1 parte 1 (PLIEGO §2 y §6): cliente caché, máscaras, estado. Sin red."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from atisbo.env.render import a_mp4, render
from atisbo.env.rutina import elegir_accion
from atisbo.estado import EstadoSim, estado_desde_mascaras
from atisbo.meta_api import CacheMiss
from atisbo.sam import (
    PROMPT,
    ClienteSam,
    SeguidorEstado,
    mascaras_por_frame,
)

FIXDIR = Path("tests/fixtures/sam")


def _sha_frames(frames) -> str:
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _episodio():
    from atisbo.lazo import Mundo

    mundo = Mundo(seed=1000, p_perturb=0.02)
    s = mundo.reiniciar()
    e = [s]
    while len(e) < 60:
        s = mundo.avanzar(s, elegir_accion(s))
        e.append(s)
    return e, [render(x) for x in e]


def _fixture_por_frames_hash(h: str) -> dict:
    for f in FIXDIR.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("payload", {}).get("frames") == h:
            return d
    raise AssertionError("sin fixture para frames %s" % h)


def _texto_real() -> str:
    e, frames = _episodio()
    h = _sha_frames([frames[0]])
    for f in FIXDIR.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("payload", {}).get("medio") == "imagen" and d.get("payload", {}).get("frames") == h:
            return str(d["texto"])
    raise AssertionError("sin fixture imagen")


def _multiset(objs):
    return sorted((t, x, y, bool(e)) for _, t, x, y, e in objs)


def test_a_mp4_60_frames_384(tmp_path):
    e, frames = _episodio()
    assert len(frames) == 60
    import imageio.v2 as imageio

    out = tmp_path / "ep.mp4"
    a_mp4(frames, out)
    rdr = imageio.get_reader(str(out))
    n = 0
    for fr in rdr:
        assert fr.shape[0] == 384 and fr.shape[1] == 384
        n += 1
    rdr.close()
    assert n == 60


def test_b_mascaras_video_60_ids_estables():
    e, frames = _episodio()
    d = _fixture_por_frames_hash(_sha_frames(frames))
    m = mascaras_por_frame(d["texto"], "video")
    assert len(m) == 60
    for fi, por_id in m.items():
        for oid, mask in por_id.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == bool
            assert mask.shape == (384, 384)
    m0 = m[0]
    bot0 = e[0].bot
    bot_id = None
    for oid, mask in m0.items():
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean() // 32), int(xs.mean() // 32)
        if (cx, cy) == bot0:
            lado = max(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)
            if lado >= 22:
                bot_id = oid
                break
    assert bot_id is not None
    ok = 0
    for fi in range(60):
        mask = m[fi].get(bot_id)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean() // 32), int(xs.mean() // 32)
        if (cx, cy) == e[fi].bot:
            ok += 1
    assert ok >= 57


def test_c_estado_frame_a_frame_y_seguidor():
    e, frames = _episodio()
    d = _fixture_por_frames_hash(_sha_frames(frames))
    m = mascaras_por_frame(d["texto"], "video")
    fallos = []
    ok = 0
    for fi in range(60):
        visto = estado_desde_mascaras(m[fi], frames[fi])
        esp = e[fi]
        got_set = set((t, x, y) for _, t, x, y, _ in visto.objetos)
        exp_set = set((t, x, y) for _, t, x, y, _ in esp.objetos)
        if visto.bot == esp.bot and got_set == exp_set:
            ok += 1
        else:
            fallos.append(fi)
    print("fallos sin seguidor:", fallos)
    assert ok >= 54
    seg = SeguidorEstado()
    for fi in range(60):
        visto = estado_desde_mascaras(m[fi], frames[fi])
        est = seg.actualizar(visto)
        esp = e[fi]
        assert est.bot == esp.bot, fi
        assert est.carga == esp.carga, fi
        assert _multiset(est.objetos) == _multiset(esp.objetos), fi
        assert est.bloqueo == esp.bloqueo, fi
        assert est.paso == fi, fi


def test_d_vacio_y_ciego():
    frame = np.zeros((384, 384, 3), dtype=np.uint8)
    v = estado_desde_mascaras({}, frame)
    assert v.bot is None
    assert tuple(v.objetos) == ()
    seg = SeguidorEstado()
    s0 = EstadoSim(bot=(3, 4), carga="rojo", objetos=((1, "rojo", 3, 4, False),),
                   cubos={"rojo": (0, 0), "verde": (11, 0), "azul": (0, 11)}, bloqueo=None, paso=0)
    r0 = seg.actualizar(s0)
    assert seg.ciego is False
    n_obj = len(r0.objetos)
    vacio = EstadoSim(bot=None, carga=None, objetos=(), cubos=dict(r0.cubos), bloqueo=None, paso=0)
    r1 = seg.actualizar(vacio)
    assert seg.ciego is True
    assert r1.bot == r0.bot
    assert r1.carga == r0.carga
    assert len(r1.objetos) == n_obj
    r2 = seg.actualizar(s0)
    assert seg.ciego is False


def test_e_replay_video_imagen_y_cachemiss():
    e, frames = _episodio()
    cli = ClienteSam(modo="replay", cache_dir=str(FIXDIR))
    txt_video = cli.segmentar(frames)
    assert txt_video == _fixture_por_frames_hash(_sha_frames(frames))["texto"]
    f0 = render(e[0])
    txt_img = cli.segmentar([f0])
    assert txt_img == _fixture_por_frames_hash(_sha_frames([f0]))["texto"]
    with pytest.raises(CacheMiss):
        cli.segmentar(frames, prompt="otro prompt que no existe")


def _sse_cuerpo(chunks, uso=None, completado=True, tipo_final="response.completed"):
    lineas = []
    for c in chunks:
        lineas.append("data: " + json.dumps({"type": "response.output_text.delta", "delta": c}))
    if completado:
        if tipo_final == "response.completed":
            lineas.append("data: " + json.dumps({"type": "response.completed", "response": {"usage": uso or {}}}))
        else:
            lineas.append("data: " + json.dumps({"type": tipo_final}))
    lineas.append("data: [DONE]")
    return ("\n".join(lineas) + "\n").encode("utf-8")


class _FakeTransporte:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []
        self.cabeceras = []
        self.cuerpos = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append(url)
        self.cabeceras.append(dict(cabeceras))
        self.cuerpos.append(cuerpo)
        idx = min(len(self.llamadas) - 1, len(self.respuestas) - 1)
        return self.respuestas[idx]


def test_e_api_graba_uso_reintentos_y_errores(tmp_path, monkeypatch):
    from atisbo.sam import MetaAPIError

    monkeypatch.setenv("META_API_KEY", "CLAVE-TEST-123")
    real = _texto_real()
    uso = {"images_processed": 1}
    fake = _FakeTransporte([(200, {}, _sse_cuerpo([real], uso))])
    cli = ClienteSam(modo="api", cache_dir=str(tmp_path), transporte=fake, dormir=lambda s: None)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    txt = cli.segmentar([frame])
    assert txt == real
    assert cli.imagenes == 1
    fichs = list(tmp_path.glob("*.json"))
    assert len(fichs) == 1
    data = json.loads(fichs[0].read_text(encoding="utf-8"))
    assert set(data.keys()) == {"payload", "texto", "uso", "latencia_ms"}
    crudo = fichs[0].read_text(encoding="utf-8")
    assert "CLAVE-TEST-123" not in crudo
    assert "base64" not in crudo
    assert "data:image" not in crudo
    cab = fake.cabeceras[0]
    assert cab["Authorization"] == "Bearer CLAVE-TEST-123"
    assert cab["Content-Type"] == "application/json"
    assert cab["Accept"] == "text/event-stream"
    body = json.loads(fake.cuerpos[0].decode("utf-8"))
    assert body["model"] == "sam-3.1"
    assert body["stream"] is True
    assert body["metadata"] == {"include_confidence": "true"}
    contenido = body["input"][0]["content"]
    assert contenido[0] == {"type": "input_text", "text": PROMPT}
    assert contenido[1]["type"] == "input_image"
    assert contenido[1]["image_url"].startswith("data:image/png;base64,")
    uso_vid = {"video_frames_processed": 3}
    fake_v = _FakeTransporte([(200, {}, _sse_cuerpo([real], uso_vid))])
    cli_v = ClienteSam(modo="api", cache_dir=str(tmp_path / "cv"), transporte=fake_v, dormir=lambda s: None)
    frames_v = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    assert cli_v.segmentar(frames_v) == real
    assert cli_v.frames_video == 3
    body_v = json.loads(fake_v.cuerpos[0].decode("utf-8"))
    contenido_v = body_v["input"][0]["content"]
    assert contenido_v[1]["type"] == "input_video"
    assert contenido_v[1]["video_url"].startswith("data:video/mp4;base64,")
    fake2 = _FakeTransporte([(429, {}, b"busy"), (200, {}, _sse_cuerpo([real], uso))])
    cli2 = ClienteSam(modo="api", cache_dir=str(tmp_path / "c2"), transporte=fake2, dormir=lambda s: None)
    assert cli2.segmentar([frame]) == real
    assert len(fake2.llamadas) == 2
    fake3 = _FakeTransporte([(400, {}, b"bad")])
    cli3 = ClienteSam(modo="api", cache_dir=str(tmp_path / "c3"), transporte=fake3, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli3.segmentar([frame])
    assert len(fake3.llamadas) == 1
    assert list((tmp_path / "c3").glob("*.json")) == []
    fake_err = _FakeTransporte([(200, {}, _sse_cuerpo([], {}, completado=True, tipo_final="error"))])
    cli_err = ClienteSam(modo="api", cache_dir=str(tmp_path / "ce"), transporte=fake_err, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli_err.segmentar([frame])
    assert list((tmp_path / "ce").glob("*.json")) == []
    fake4 = _FakeTransporte([(200, {}, _sse_cuerpo([], {}, completado=True, tipo_final="response.failed"))])
    cli4 = ClienteSam(modo="api", cache_dir=str(tmp_path / "c4"), transporte=fake4, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli4.segmentar([frame])
    assert list((tmp_path / "c4").glob("*.json")) == []
    fake5 = _FakeTransporte([(200, {}, _sse_cuerpo([real], uso, completado=False))])
    cli5 = ClienteSam(modo="api", cache_dir=str(tmp_path / "c5"), transporte=fake5, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli5.segmentar([frame])
    assert list((tmp_path / "c5").glob("*.json")) == []
    fake6 = _FakeTransporte([(200, {}, _sse_cuerpo([], uso))])
    cli6 = ClienteSam(modo="api", cache_dir=str(tmp_path / "c6"), transporte=fake6, dormir=lambda s: None)
    assert cli6.segmentar([frame]) == ""
    fake_nj = _FakeTransporte([(200, {}, b"data: no-es-json\n" + _sse_cuerpo([real], uso))])
    cli_nj = ClienteSam(modo="api", cache_dir=str(tmp_path / "cnj"), transporte=fake_nj, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli_nj.segmentar([frame])
    assert list((tmp_path / "cnj").glob("*.json")) == []
    monkeypatch.delenv("META_API_KEY", raising=False)
    fake7 = _FakeTransporte([(200, {}, _sse_cuerpo([real], uso))])
    cli7 = ClienteSam(modo="api", cache_dir=str(tmp_path / "c7"), transporte=fake7, dormir=lambda s: None)
    with pytest.raises(MetaAPIError):
        cli7.segmentar([frame])


def test_f_imagenes_bot_carga_bloqueo():
    e, frames = _episodio()
    idxs = [0, 7, 20, 30, 45, 59]
    for fi in idxs:
        h = _sha_frames([frames[fi]])
        d = _fixture_por_frames_hash(h)
        m = mascaras_por_frame(d["texto"], "imagen")
        visto = estado_desde_mascaras(m[0], frames[fi])
        esp = e[fi]
        assert visto.bot == esp.bot, fi
        assert visto.carga == esp.carga, fi
        assert visto.bloqueo == esp.bloqueo, fi
    for fi, bl in [(12, (5, 5)), (33, (9, 3))]:
        fr = render(dataclasses.replace(e[fi], bloqueo=bl))
        d = _fixture_por_frames_hash(_sha_frames([fr]))
        m = mascaras_por_frame(d["texto"], "imagen")
        visto = estado_desde_mascaras(m[0], fr)
        assert visto.bot == e[fi].bot
        assert visto.bloqueo == bl
        assert visto.carga == e[fi].carga
        assert set((t, x, y) for _, t, x, y, _ in visto.objetos) == set(
            (t, x, y) for _, t, x, y, _ in e[fi].objetos
        )
    for fi in [0, 20, 30, 59]:
        h = _sha_frames([frames[fi]])
        d = _fixture_por_frames_hash(h)
        m = mascaras_por_frame(d["texto"], "imagen")
        visto = estado_desde_mascaras(m[0], frames[fi])
        assert set((t, x, y) for _, t, x, y, _ in visto.objetos) == set(
            (t, x, y) for _, t, x, y, _ in e[fi].objetos
        ), fi
    for fi in [7, 45]:
        h = _sha_frames([frames[fi]])
        d = _fixture_por_frames_hash(h)
        m = mascaras_por_frame(d["texto"], "imagen")
        visto = estado_desde_mascaras(m[0], frames[fi])
        exp_set = set((t, x, y) for _, t, x, y, _ in e[fi].objetos)
        got_set = set((t, x, y) for _, t, x, y, _ in visto.objetos)
        assert got_set < exp_set
        assert len(exp_set - got_set) == 1


def test_g_contadores_y_coste():
    e, frames = _episodio()
    cli = ClienteSam(modo="replay", cache_dir=str(FIXDIR))
    dvid = _fixture_por_frames_hash(_sha_frames(frames))
    cli.segmentar(frames)
    assert cli.frames_video == dvid["uso"].get("video_frames_processed", 60)
    f0 = render(e[0])
    dimg = _fixture_por_frames_hash(_sha_frames([f0]))
    cli.segmentar([f0])
    assert cli.imagenes == dimg["uso"].get("images_processed", 1)
    assert abs(cli.coste_estimado_usd() - (cli.imagenes * 0.0025 + cli.frames_video * 0.0002)) < 1e-12


def test_h_barajado_y_faltantes():
    _e, frames = _episodio()
    d = _fixture_por_frames_hash(_sha_frames(frames))
    m = mascaras_por_frame(d["texto"], "video")
    lineas = d["texto"].splitlines()
    rng = random.Random(0)
    rng.shuffle(lineas)
    quitados = {5, 17}
    recortadas = []
    for ln in lineas:
        if ln.startswith("<"):
            fi = int(ln[1:].split("f>")[0])
            if fi in quitados:
                continue
        recortadas.append(ln)
    m2 = mascaras_por_frame("\n".join(recortadas) + "\n", "video")
    assert set(m2.keys()) == set(m.keys()) - quitados
    for fi, por_id in m.items():
        if fi in quitados:
            continue
        assert set(m2[fi].keys()) == set(por_id.keys())
        for oid in por_id:
            assert np.array_equal(m2[fi][oid], por_id[oid])
