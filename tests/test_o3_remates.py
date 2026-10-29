"""Remates O3 parte 1: lazo y sorpresa (r1-r10). Sin red."""
from __future__ import annotations

import random

import numpy as np
import pytest

from atisbo.control import Prediccion, PredictorNulo, PredictorOraculo, PredictorRuido
from atisbo.env.mundo import es_exito
from atisbo.env.perturbaciones import aplicar_mover
from atisbo.env.trazas import generar
from atisbo.estado import Accion, vector
from atisbo.lazo import Mundo, correr_episodio, metricas_desde_registro
from atisbo.politica_llm import PoliticaSimulada
from atisbo.sorpresa import errores_a_un_paso, sorpresa


class _MundoMueveTras4(Mundo):
    """Mueve un objeto tras la 4.ª acción (sorpresa en j=1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.movido = False

    def avanzar(self, s, a):
        s2 = super().avanzar(s, a)
        if self._pasos == 4 and not es_exito(s2):
            s3, ok = aplicar_mover(s2, random.Random(0))
            if ok:
                self.movido = True
                self._cur = s3
                return s3
        return s2


class _PredEspera:
    """Siempre ESPERAR con conf 1; estados = último vector repetido."""

    def predecir(self, ventana):
        arr = np.asarray(ventana, dtype=np.float32)
        ultimo = arr[-1].astype(np.float32)
        return Prediccion(
            acciones=(Accion.ESPERAR, Accion.ESPERAR, Accion.ESPERAR),
            estados=np.stack([ultimo, ultimo, ultimo]).astype(np.float32),
            confianza=1.0,
        )


class _TruncaPrimera:
    """Política de prueba que se trunca en su primera decisión."""

    def __init__(self):
        self.llm_invalidas = 0
        self.peticiones = 0
        self.llamadas_red = 0
        self.truncada = False

    def decidir(self, estado):
        self.truncada = True
        return Accion.ESPERAR, 0.0


def test_r1_ejecutado_es_registrado():
    mundo = Mundo(seed=0, p_perturb=0.05)
    predictor = PredictorRuido(0.3, seed=7)
    aplicadas: list = []
    orig = mundo.avanzar

    def envuelto(s, a):
        aplicadas.append(a.name)
        return orig(s, a)

    mundo.avanzar = envuelto
    for i in range(3):
        aplicadas.clear()
        pol = PoliticaSimulada(seed=100 + i, p=0.05)
        registro: list = []
        res = correr_episodio(predictor, pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
        assert res["pasos"] == len(registro)
        assert aplicadas == [e["accion"] for e in registro]


def test_r2_sorpresa_en_j1_corta_bloque():
    mundo = _MundoMueveTras4(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    predictor = PredictorOraculo()
    llamadas_en_len: list = []
    registro_ref: list = []
    orig = predictor.predecir

    def contado(ventana):
        llamadas_en_len.append(len(registro_ref))
        return orig(ventana)

    predictor.predecir = contado
    res = correr_episodio(predictor, pol, mundo, tau=0.5, epsilon=0.01, registro=registro_ref)
    assert mundo.movido is True
    assert len(registro_ref) >= 6
    assert registro_ref[4]["fuente"] == "llm"
    assert float(registro_ref[4]["sorpresa"]) > 0.01
    assert registro_ref[5]["fuente"] == "atisbo"
    assert 5 in llamadas_en_len


def test_r3_truncado_dentro_del_bloque():
    mundo = _MundoMueveTras4(seed=0, p_perturb=0.0)
    pol = _TruncaPrimera()
    registro: list = []
    res = correr_episodio(PredictorOraculo(), pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
    assert mundo.movido is True
    assert res["truncada"] is True
    assert res["pasos"] == len(registro)
    assert all(e["fuente"] != "llm" for e in registro)


def test_r4_presupuesto_dentro_del_bloque():
    mundo = Mundo(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    registro: list = []
    res = correr_episodio(_PredEspera(), pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
    assert res["pasos"] == 160
    assert len(registro) == 160


def test_r5_exito_dentro_del_bloque():
    esperados = generar(0, 20, 0.0)
    mundo = Mundo(seed=0, p_perturb=0.0)
    for i in range(20):
        pol = PoliticaSimulada(seed=0)
        registro: list = []
        res = correr_episodio(PredictorOraculo(), pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
        assert res["pasos"] == len(esperados[i])
        assert res["exito"] is True


def test_r6_bordes():
    mundo = Mundo(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    registro: list = []
    correr_episodio(PredictorOraculo(), pol, mundo, tau=1.0, epsilon=0.01, registro=registro)
    assert len(registro) > 0
    assert registro[0]["fuente"] == "atisbo"

    mundo_a = _MundoMueveTras4(seed=0, p_perturb=0.0)
    reg_a: list = []
    correr_episodio(PredictorOraculo(), PoliticaSimulada(seed=0), mundo_a, tau=0.5, epsilon=0.01, registro=reg_a)
    assert mundo_a.movido is True
    sorp_val = float(reg_a[4]["sorpresa"])
    assert sorp_val > 0.01

    mundo_b = _MundoMueveTras4(seed=0, p_perturb=0.0)
    reg_b: list = []
    correr_episodio(PredictorOraculo(), PoliticaSimulada(seed=0), mundo_b, tau=0.5, epsilon=sorp_val, registro=reg_b)
    assert mundo_b.movido is True
    assert len(reg_b) >= 5
    assert reg_b[4]["fuente"] == "atisbo"


def test_r7_errores_oraculo_cero():
    episodios = generar(0, 5, 0.0)
    errores = errores_a_un_paso(PredictorOraculo(), episodios)
    assert len(errores) > 0
    assert all(float(e) == pytest.approx(0.0) for e in errores)


def test_r8_sorpresa_formas_distintas():
    episodios = generar(0, 1, 0.0)
    s = episodios[0][0][0]
    with pytest.raises(ValueError):
        sorpresa(s, np.zeros(1, dtype=np.float32))


def test_r9_registro_precargado():
    precarg = [
        {"fuente": "llm", "conf": 0.0, "sorpresa": 0.0, "latencia_ms": 999.0, "accion": "ESPERAR", "accion_rutina": "DER"},
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "IZQ", "accion_rutina": "DER"},
    ]
    n0 = len(precarg)
    mundo = Mundo(seed=0, p_perturb=0.05)
    pol = PoliticaSimulada(seed=99, p=0.05)
    registro: list = list(precarg)
    res = correr_episodio(PredictorRuido(0.3, seed=7), pol, mundo, tau=0.5, epsilon=0.01, registro=registro)
    assert len(registro) == n0 + res["pasos"]
    epis = registro[n0:]
    llamadas_esp = sum(1 for e in epis if e.get("fuente") == "llm")
    malas_esp = sum(1 for e in epis if e.get("fuente") == "atisbo" and e.get("accion") != e.get("accion_rutina"))
    assert res["llamadas_llm"] == llamadas_esp
    assert res["malas_aceptadas"] == malas_esp
    assert res["tasa_malas_aceptadas"] == pytest.approx(malas_esp / res["pasos"])
    assert 999.0 not in list(res["latencias_ms"])


def test_r10_llamadas_por_sorpresa():
    mundo = _MundoMueveTras4(seed=0, p_perturb=0.0)
    reg: list = []
    correr_episodio(PredictorOraculo(), PoliticaSimulada(seed=0), mundo, tau=0.5, epsilon=0.01, registro=reg)
    m = metricas_desde_registro(reg)
    assert m["llamadas_por_sorpresa"] == 1

    mundo2 = Mundo(seed=0, p_perturb=0.0)
    reg2: list = []
    correr_episodio(PredictorNulo(), PoliticaSimulada(seed=0), mundo2, tau=0.5, epsilon=1e9, registro=reg2)
    m2 = metricas_desde_registro(reg2)
    assert m2["llamadas_por_sorpresa"] == 0


def test_r11_ultima_racha_atisbo():
    reg = [
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "ESPERAR", "accion_rutina": "ESPERAR"},
        {"fuente": "llm", "conf": 0.0, "sorpresa": 0.0, "latencia_ms": 1.0, "accion": "ESPERAR", "accion_rutina": "ESPERAR"},
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "ESPERAR", "accion_rutina": "ESPERAR"},
        {"fuente": "atisbo", "conf": 1.0, "sorpresa": 0.0, "latencia_ms": 0.0, "accion": "ESPERAR", "accion_rutina": "ESPERAR"},
    ]
    m = metricas_desde_registro(reg)
    assert m["racha_media"] == pytest.approx(1.5)
