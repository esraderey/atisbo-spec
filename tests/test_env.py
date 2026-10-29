import dataclasses
import pickle

import imageio.v2 as imageio
import numpy as np
import pytest

from atisbo.env.mundo import aplicar, es_exito
from atisbo.env.render import BOT_COLOR, a_mp4, render
from atisbo.env.rutina import elegir_accion
from atisbo.env.trazas import generar
from atisbo.estado import Accion, EstadoSim, vector


def _cubos():
    return {"rojo": (0, 0), "verde": (11, 0), "azul": (0, 11)}


def test_accion_orden_y_estado_congelado():
    assert [a.name for a in Accion] == ["ARRIBA", "ABAJO", "IZQ", "DER", "TOMAR", "SOLTAR", "ESPERAR"]
    s = EstadoSim(bot=(5, 5), carga=None, objetos=((1, "rojo", 5, 5, False),),
                  cubos=_cubos(), bloqueo=None, paso=0)
    hash(s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.paso = 1  # type: ignore[misc] # EstadoSim es frozen: el ignore calla al checker, pytest.raises comprueba el error


def test_a_determinismo_pickle():
    ep1 = generar(7, 5, 0.02)
    ep2 = generar(7, 5, 0.02)
    assert pickle.dumps(ep1) == pickle.dumps(ep2)
    ep3 = generar(8, 5, 0.02)
    assert pickle.dumps(ep1) != pickle.dumps(ep3)


def test_b_rutina_resuelve():
    eps = generar(0, 50, 0.0)
    assert len(eps) == 50
    for ep in eps:
        assert len(ep) <= 120
        assert es_exito(aplicar(*ep[-1]))
    eps_p = generar(1, 50, 0.05)
    ok = sum(1 for ep in eps_p if es_exito(aplicar(*ep[-1])))
    assert ok >= 40


def test_c_tomar_sin_objeto_no_cambia():
    s = EstadoSim(bot=(5, 5), carga=None, objetos=((1, "rojo", 7, 7, False),),
                  cubos=_cubos(), bloqueo=None, paso=3)
    s2 = aplicar(s, Accion.TOMAR)
    assert s2.bot == s.bot and s2.carga is None and s2.objetos == s.objetos


def test_c_soltar_fuera_deja_en_celda():
    s = EstadoSim(bot=(5, 5), carga="rojo", objetos=(),
                  cubos=_cubos(), bloqueo=None, paso=0)
    s2 = aplicar(s, Accion.SOLTAR)
    assert s2.carga is None
    assert len(s2.objetos) == 1
    _id, tipo, x, y, en_cubo = s2.objetos[0]
    assert (tipo, x, y, en_cubo) == ("rojo", 5, 5, False)


def test_c_moverse_borde_y_bloqueo_es_quedarse():
    s = EstadoSim(bot=(0, 0), carga=None, objetos=(),
                  cubos=_cubos(), bloqueo=None, paso=0)
    assert aplicar(s, Accion.IZQ).bot == (0, 0)
    assert aplicar(s, Accion.ARRIBA).bot == (0, 0)
    sb = EstadoSim(bot=(5, 5), carga=None, objetos=(),
                   cubos=_cubos(), bloqueo=(6, 5), paso=0)
    assert aplicar(sb, Accion.DER).bot == (5, 5)


def test_c_desempate_por_id_menor():
    s = EstadoSim(bot=(6, 6), carga=None,
                  objetos=((5, "rojo", 7, 6, False), (2, "verde", 6, 7, False)),
                  cubos=_cubos(), bloqueo=None, paso=0)
    assert elegir_accion(s) == Accion.ABAJO


def test_d_vector_dimension_y_rango():
    s3 = EstadoSim(bot=(6, 6), carga=None,
                   objetos=((1, "rojo", 1, 1, False), (2, "verde", 2, 2, True), (3, "azul", 3, 3, False)),
                   cubos=_cubos(), bloqueo=None, paso=0)
    s6 = EstadoSim(bot=(6, 6), carga="rojo",
                   objetos=tuple((i, "rojo", i, i, False) for i in range(1, 7)),
                   cubos=_cubos(), bloqueo=(4, 4), paso=0)
    v3, v6 = vector(s3), vector(s6)
    assert v3.shape == v6.shape
    assert v3.dtype == np.float32
    assert float(v3.min()) >= 0.0 and float(v3.max()) <= 1.0
    assert float(v6.min()) >= 0.0 and float(v6.max()) <= 1.0


def test_e_render_y_mp4(tmp_path):
    s = EstadoSim(bot=(5, 5), carga=None, objetos=((1, "rojo", 1, 1, False),),
                  cubos=_cubos(), bloqueo=None, paso=0)
    img = render(s)
    assert img.shape == (384, 384, 3) and img.dtype == np.uint8
    cx, cy = 5 * 32 + 16, 5 * 32 + 16
    assert tuple(img[cy, cx].tolist()) == tuple(BOT_COLOR)
    ep = generar(0, 2, 0.0)[0]
    frames = [render(s_) for s_, _ in ep]
    out = tmp_path / "e.mp4"
    a_mp4(frames, out)
    assert out.stat().st_size > 1024
    assert imageio.get_reader(str(out)).count_frames() == len(frames)


def test_f_perturbaciones_registradas():
    eps = generar(0, 200, 0.05)
    vistos = set()
    for ep in eps:
        for _paso, tipo in ep.perturbaciones:
            vistos.add(tipo)
            assert isinstance(_paso, int)
    assert {"mover", "anadir", "bloqueo"} <= vistos
