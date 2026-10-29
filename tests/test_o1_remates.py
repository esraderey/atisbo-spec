import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atisbo.baselines import ngram3, azar
from atisbo.control import PredictorOraculo
from atisbo.env.mundo import aplicar
from atisbo.env.trazas import Episodio, estado_inicial
from atisbo.estado import Accion, vector
from atisbo.exp import comun
from atisbo.exp import e0 as exp_e0
from atisbo.exp.comun import bootstrap, commit_hash, escribir_json, validar
from atisbo.metrics import acierto_at_1, acierto_at_3
from atisbo.modelo import Atisbo


def _eps_uniformes(seed, n_ep, n_pasos=60):
    rng_ini = random.Random(f"{seed}:ini")
    rng_act = random.Random(seed)
    acts = list(Accion)
    eps = []
    for _ in range(n_ep):
        s = estado_inicial(rng_ini)
        pasos = []
        for _ in range(n_pasos):
            a = rng_act.choice(acts)
            pasos.append((s, a))
            s = aplicar(s, a)
        eps.append(Episodio(pasos, perturbaciones=(), exito=False))
    return eps


@pytest.fixture(scope="module")
def datos_rapido(tmp_path_factory):
    out = tmp_path_factory.mktemp("e0rap") / "E0.json"
    r = subprocess.run(
        [sys.executable, "-m", "atisbo.exp.e0", "--rapido", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    return json.loads(out.read_text(encoding="utf-8"))


def _esquema():
    return json.loads(Path("atisbo/exp/schema_results.json").read_text(encoding="utf-8"))


def test_e_arnes_real_uniforme():
    tr_eps = _eps_uniformes(11, 20, 60)
    te_eps = _eps_uniformes(12, 10, 60)
    m = Atisbo.entrenar(tr_eps, seed=0, epocas=2)
    ng = ngram3().ajustar([[a for _, a in ep] for ep in tr_eps])
    az = azar(seed=0)
    limite1 = 1.0 / 7.0 + 0.05
    limite3 = 0.05
    for metodo in ("atisbo", "repetir_ultima", "ngram3", "azar"):
        tot_ac1 = 0.0
        tot_ac3 = 0.0
        tot_n = 0
        for ep in te_eps:
            r = exp_e0._eval_episodio(ep, m, ng, az)
            assert r is not None
            n = len(ep) - 3
            tot_ac1 += r[metodo]["acc1"] * n
            tot_ac3 += r[metodo]["acc3"] * n
            tot_n += n
        assert tot_n > 0
        assert tot_ac1 / tot_n <= limite1, metodo
        assert tot_ac3 / tot_n <= limite3, metodo


def test_c_ventana_real():
    eps = _eps_uniformes(21, 2, 20)
    ep = eps[0]
    k = 8
    d = len(vector(ep[0][0]))
    v0 = comun.ventana(ep, 0, k)
    assert v0.shape == (k, d)
    v_first = vector(ep[0][0])
    for j in range(k):
        np.testing.assert_array_equal(v0[j], v_first)
    t = 5
    vt = comun.ventana(ep, t, k)
    np.testing.assert_array_equal(vt[-1], vector(ep[t][0]))
    assert vt.shape == (k, d)


def test_g_schema_ok(datos_rapido):
    validar(datos_rapido, _esquema())


def test_g_rapido_n_igual_4(datos_rapido):
    assert len(datos_rapido["metricas"]) > 0
    for clave, met in datos_rapido["metricas"].items():
        assert met["n"] == 4, clave
        assert met["n_remuestreos"] == 1000


def test_g_rechaza_veredicto(datos_rapido):
    datos = copy.deepcopy(datos_rapido)
    datos["veredicto"] = "quizas"
    with pytest.raises(ValueError):
        validar(datos, _esquema())


def test_g_rechaza_n_parametros(datos_rapido):
    datos = copy.deepcopy(datos_rapido)
    datos["entrenamiento"]["n_parametros"] = 2000000
    with pytest.raises(ValueError):
        validar(datos, _esquema())


def test_g_rechaza_clave_extra(datos_rapido):
    datos = copy.deepcopy(datos_rapido)
    primera = next(iter(datos["metricas"]))
    datos["metricas"][primera]["clave_de_mas"] = 1
    with pytest.raises(ValueError):
        validar(datos, _esquema())


def test_g_rechaza_nan_media(datos_rapido):
    datos = copy.deepcopy(datos_rapido)
    primera = next(iter(datos["metricas"]))
    datos["metricas"][primera]["media"] = float("nan")
    with pytest.raises(ValueError):
        validar(datos, _esquema())


def test_g_rechaza_palabra_no_soportada(datos_rapido):
    from atisbo.exp.comun import SchemaError
    with pytest.raises(SchemaError):
        validar(datos_rapido, {"palabra_no_soportada": 1})


def test_g_rechaza_tipo_desconocido():
    from atisbo.exp.comun import SchemaError
    with pytest.raises(SchemaError):
        validar(1, {"type": "desconocido"})


def test_g_const_enum_estricto():
    with pytest.raises(ValueError):
        validar(True, {"const": 1})
    with pytest.raises(ValueError):
        validar(1, {"const": True})
    with pytest.raises(ValueError):
        validar(1, {"enum": [True, False]})
    with pytest.raises(ValueError):
        validar(True, {"enum": [1, 2]})
    validar(1, {"const": 1})
    validar(True, {"const": True})


def test_g_number_sin_nan_inf():
    for malo in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            validar({"m": malo}, {"type": "object", "properties": {"m": {"type": "number"}}})
    validar({"m": 0.5}, {"type": "object", "properties": {"m": {"type": "number"}}})


def test_g_escribir_json_sin_nan(tmp_path):
    with pytest.raises(ValueError):
        escribir_json(tmp_path / "m.json", {"media": float("nan")})


def test_g_llm_obligatorio_e1(datos_rapido):
    datos = copy.deepcopy(datos_rapido)
    datos["experimento"] = "E1"
    datos.pop("llm", None)
    with pytest.raises(ValueError):
        validar(datos, _esquema())
    datos["llm"] = {"modo": "simulada"}
    validar(datos, _esquema())


def test_d_oraculo_acierta_3():
    eps = comun.generar_trazas(3, 3, 0.0)
    ora = PredictorOraculo()
    checked = 0
    for ep in eps:
        assert len(ep) >= 4
        for t in range(len(ep) - 3):
            v = comun.ventana(ep, t)
            p = ora.predecir(v)
            verdad = [ep[t][1], ep[t + 1][1], ep[t + 2][1]]
            assert list(p.acciones) == list(verdad)
            esperado = np.stack(
                [vector(ep[t + 1][0]), vector(ep[t + 2][0]), vector(ep[t + 3][0])]
            ).astype(np.float32)
            np.testing.assert_array_equal(np.asarray(p.estados), esperado)
            assert float(p.confianza) == 1.0
            checked += 1
    assert checked > 0


def test_acierto_at_3_exige_3():
    a = [Accion.ARRIBA, Accion.ABAJO, Accion.IZQ]
    b = [Accion.ARRIBA, Accion.ABAJO, Accion.DER]
    assert acierto_at_3(a, a) == 1.0
    assert acierto_at_3(a, b) == 0.0
    assert acierto_at_1(a, b) == 1.0


def test_regla_go_no_go():
    from atisbo.exp.e0 import decidir_veredicto
    ps = [0.0]
    base_go = {
        "atisbo/acierto@3/p=0.0": {"media": 0.9, "ic95": [0.8, 0.95]},
        "repetir_ultima/acierto@3/p=0.0": {"media": 0.2, "ic95": [0.1, 0.3]},
        "ngram3/acierto@3/p=0.0": {"media": 0.25, "ic95": [0.15, 0.35]},
        "azar/acierto@3/p=0.0": {"media": 0.01, "ic95": [0.0, 0.05]},
    }
    assert decidir_veredicto(base_go, ps) == "go"
    base_frontera = dict(base_go)
    base_frontera["atisbo/acierto@3/p=0.0"] = {"media": 0.32, "ic95": [0.25, 0.34]}
    assert decidir_veredicto(base_frontera, ps) == "no-go"
    base_nogo = dict(base_go)
    base_nogo["atisbo/acierto@3/p=0.0"] = {"media": 0.2, "ic95": [0.1, 0.3]}
    assert decidir_veredicto(base_nogo, ps) == "no-go"


def test_bootstrap_anchura():
    vals = [0.0] * 50 + [1.0] * 50
    b = bootstrap(vals, n_remuestreos=1000, seed=0)
    assert b["media"] == pytest.approx(0.5)
    ancho = b["ic95"][1] - b["ic95"][0]
    assert 0.15 < ancho < 0.25


def test_ngram_desempate_orden_a():
    ng = ngram3().ajustar(
        [[Accion.ARRIBA, Accion.ABAJO, Accion.DER], [Accion.ARRIBA, Accion.ABAJO, Accion.IZQ]]
    )
    pred = ng.predecir([Accion.ARRIBA, Accion.ABAJO])
    assert pred[0] == Accion.IZQ
    ng2 = ngram3().ajustar([[Accion.ABAJO, Accion.ARRIBA]])
    assert ng2.predecir([])[0] == Accion.ARRIBA


def test_commit_hash_marca_sucio():
    h = commit_hash()
    try:
        r1 = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        assert h == "sin-git"
        return
    head = r1.stdout.strip()
    if r1.returncode != 0 or not head:
        assert h == "sin-git"
        return
    try:
        r2 = subprocess.run(
            ["git", "status", "--porcelain", "--", "atisbo", "pyproject.toml"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        assert h == head + "+sucio"
        return
    if r2.returncode != 0:
        assert h == head + "+sucio"
        return
    sucio = bool(r2.stdout.strip())
    esperado = head + ("+sucio" if sucio else "")
    assert h == esperado


def test_confianza_es_producto():
    import torch

    tr_eps = _eps_uniformes(31, 10, 30)
    m = Atisbo.entrenar(tr_eps, seed=0, epocas=2)
    ep = tr_eps[0]
    v = comun.ventana(ep, 0)
    p = m.predecir(v)
    m._red.eval()
    with torch.no_grad():
        xt = torch.from_numpy(np.asarray(v, dtype=np.float32)[np.newaxis, :, :])
        logits, _ = m._red(xt)
        probs = torch.softmax(logits[0], dim=-1).numpy()
        maximos = [float(np.max(probs[j])) for j in range(3)]
        producto = float(np.prod(maximos))
        suma = float(np.sum(maximos))
    assert p.confianza == pytest.approx(producto)
    assert abs(suma - producto) > 0.1
    assert abs(p.confianza - suma) > 1e-6


def test_semilla_sin_episodios_error():
    from atisbo.exp.e0 import promedio_por_seed
    with pytest.raises(ValueError):
        promedio_por_seed([])
    assert promedio_por_seed([0.5, 0.5]) == pytest.approx(0.5)
