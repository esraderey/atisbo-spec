"""Anclas O3 primera parte: sorpresa y lazo especulativo (PLIEGO §5). Sin red."""
from __future__ import annotations

import random

import numpy as np
import pytest

from atisbo.control import PredictorNulo, PredictorOraculo, PredictorRuido
from atisbo.env.mundo import es_exito
from atisbo.env.perturbaciones import aplicar_mover
from atisbo.env.trazas import generar
from atisbo.exp.comun import ventana
from atisbo.estado import vector
from atisbo.lazo import Mundo, correr_episodio, metricas_desde_registro
from atisbo.politica_llm import PoliticaSimulada
from atisbo.sorpresa import calibrar_epsilon, errores_a_un_paso, sorpresa


def test_a_oraculo_sin_llamadas_y_exito():
    mundo = Mundo(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=123)
    for _ in range(20):
        registro: list = []
        res = correr_episodio(PredictorOraculo(), pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
        assert res["llamadas_llm"] == 0
        assert res["exito"] is True
        assert res["pasos"] > 0
        assert len(registro) == res["pasos"]
        assert all(e["fuente"] == "atisbo" for e in registro)


def test_b_nulo_todo_llm():
    mundo = Mundo(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    registro: list = []
    res = correr_episodio(PredictorNulo(), pol, mundo, tau=0.5, epsilon=1e9, registro=registro)
    assert res["pasos"] > 0
    assert res["llamadas_llm"] == res["pasos"]
    assert len(registro) == res["pasos"]
    assert all(e["fuente"] == "llm" for e in registro)


def test_c_ruido_malas_solo_atisbo():
    predictor = PredictorRuido(0.3, seed=7)
    mundo = Mundo(seed=0, p_perturb=0.05)
    pol = PoliticaSimulada(seed=99)
    total_llamadas = 0
    total_pasos = 0
    total_malas = 0
    todo: list = []
    for _ in range(5):
        registro: list = []
        res = correr_episodio(predictor, pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
        total_llamadas += int(res["llamadas_llm"])
        total_pasos += int(res["pasos"])
        total_malas += int(res["malas_aceptadas"])
        todo.extend(registro)
    assert 0 < total_llamadas < total_pasos
    esperadas = sum(1 for e in todo if e["fuente"] == "atisbo" and e["accion"] != e["accion_rutina"])
    assert total_malas == esperadas
    assert any(e["fuente"] == "llm" and e["accion"] != e["accion_rutina"] for e in todo)


def test_d_sorpresa_corta_bloque():
    class MundoMueve(Mundo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.movido = False

        def avanzar(self, s, a):
            s2 = super().avanzar(s, a)
            if self._pasos == 5 and not es_exito(s2):
                s3, ok = aplicar_mover(s2, random.Random(0))
                if ok:
                    self.movido = True
                    self._cur = s3
                    return s3
            return s2

    mundo = MundoMueve(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    registro: list = []
    correr_episodio(PredictorOraculo(), pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
    assert mundo.movido is True
    assert len(registro) >= 7
    assert all(e["fuente"] == "atisbo" for e in registro[:5])
    assert registro[5]["fuente"] == "llm"
    assert float(registro[5]["sorpresa"]) > 0.01
    assert registro[6]["fuente"] == "atisbo"

    mundo2 = MundoMueve(seed=0, p_perturb=0.0)
    pol2 = PoliticaSimulada(seed=0)
    registro2: list = []
    correr_episodio(PredictorOraculo(), pol2, mundo2, tau=0.5, epsilon=1e9, registro=registro2)
    assert mundo2.movido is True
    assert len(registro2) >= 6
    assert registro2[5]["fuente"] == "atisbo"


def test_e_racha_media_sintetica():
    registro = [
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "DER", "accion_rutina": "DER"},
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "IZQ", "accion_rutina": "DER"},
        {"fuente": "llm", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 1.0, "accion": "IZQ", "accion_rutina": "DER"},
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "DER", "accion_rutina": "DER"},
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "DER", "accion_rutina": "DER"},
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "ARRIBA", "accion_rutina": "DER"},
        {"fuente": "llm", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 2.0, "accion": "ARRIBA", "accion_rutina": "DER"},
    ]
    m = metricas_desde_registro(registro)
    assert m["racha_media"] == pytest.approx(2.5)
    assert m["malas_aceptadas"] == 2
    assert m["tasa_malas_aceptadas"] == pytest.approx(2 / 7)


def test_f_calibrar_epsilon_rango():
    episodios = generar(0, 5, 0.0)
    errores = errores_a_un_paso(PredictorRuido(0.3, seed=0), episodios)
    assert len(errores) > 0
    eps = calibrar_epsilon(PredictorRuido(0.3, seed=0), episodios)
    assert float(eps) > 0.0
    assert float(eps) <= float(max(errores))
    assert float(eps) == pytest.approx(float(np.percentile(np.asarray(errores, dtype=np.float64), 95)))


def test_sorpresa_es_l2():
    episodios = generar(0, 1, 0.0)
    s = episodios[0][0][0]
    v = vector(s)
    assert float(sorpresa(s, v)) == pytest.approx(0.0)
    delta = np.zeros_like(v)
    delta[0] = 3.0
    delta[1] = 4.0
    assert float(sorpresa(s, v + delta)) == pytest.approx(5.0)
    pred = PredictorOraculo().predecir(ventana(episodios[0], 0))
    assert float(sorpresa(episodios[0][1][0], pred.estados[0])) == pytest.approx(0.0)


def test_nan_confianza_falla_a_llm():
    import numpy as np

    from atisbo.control import Prediccion
    from atisbo.estado import Accion

    class PredConfNan:
        def predecir(self, ventana):
            d = int(np.asarray(ventana).shape[1])
            return Prediccion(
                acciones=(Accion.ESPERAR, Accion.ESPERAR, Accion.ESPERAR),
                estados=np.zeros((3, d), dtype=np.float32),
                confianza=float("nan"),
            )

    mundo = Mundo(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    registro: list = []
    res = correr_episodio(PredConfNan(), pol, mundo, tau=0.5, epsilon=1e9, registro=registro)
    assert res["pasos"] > 0
    assert res["llamadas_llm"] == res["pasos"]
    assert len(registro) == res["pasos"]
    assert all(e["fuente"] == "llm" for e in registro)


def test_nan_sorpresa_corta_bloque():
    import numpy as np

    from atisbo.control import Prediccion
    from atisbo.estado import Accion

    class PredEstadosNan:
        def predecir(self, ventana):
            d = int(np.asarray(ventana).shape[1])
            return Prediccion(
                acciones=(Accion.ESPERAR, Accion.ESPERAR, Accion.ESPERAR),
                estados=np.full((3, d), float("nan"), dtype=np.float32),
                confianza=1.0,
            )

    mundo = Mundo(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    registro: list = []
    correr_episodio(PredEstadosNan(), pol, mundo, tau=0.5, epsilon=1e9, registro=registro)
    assert len(registro) >= 2
    assert registro[0]["fuente"] == "atisbo"
    assert registro[1]["fuente"] == "llm"
    assert bool(np.isnan(float(registro[1]["sorpresa"])))
