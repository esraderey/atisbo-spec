import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from atisbo.baselines import repetir_ultima, ngram3, azar
from atisbo.env.mundo import aplicar
from atisbo.env.trazas import Episodio, estado_inicial
from atisbo.estado import Accion, vector
from atisbo.exp.comun import bootstrap, validar
from atisbo.modelo import Atisbo


def _eps_uniformes(seed, n_ep, n_pasos=100):
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


def _ventana(ep, idx, k=8):
    n = len(ep)
    vecs = []
    for j in range(k):
        ii = max(0, idx - (k - 1 - j))
        ii = min(ii, n - 1)
        vecs.append(vector(ep[ii][0]))
    return np.stack(vecs).astype(np.float32)


def _acc1_atisbo(modelo, episodios):
    tot = 0
    ok = 0
    for ep in episodios:
        if len(ep) < 4:
            continue
        for i in range(len(ep) - 3):
            p = modelo.predecir(_ventana(ep, i))
            tot += 1
            if p.acciones[0] == ep[i][1]:
                ok += 1
    return ok / tot if tot else 0.0


def _media_acc1_repetir(seq):
    ok = 0
    tot = 0
    for i in range(1, len(seq)):
        pred = repetir_ultima(seq[:i])
        tot += 1
        if pred[0] == seq[i]:
            ok += 1
    return ok / tot if tot else 0.0


def _media_acc1_ngram(tr, te):
    ng = ngram3()
    ng.ajustar([tr])
    ok = 0
    tot = 0
    for i in range(1, len(te)):
        pred = ng.predecir(te[:i])
        tot += 1
        if pred[0] == te[i]:
            ok += 1
    return ok / tot if tot else 0.0


def _media_acc1_azar(te, seed=0):
    az = azar(seed=seed)
    ok = 0
    tot = 0
    for i in range(len(te)):
        pred = az.predecir(te[:i])
        tot += 1
        if pred[0] == te[i]:
            ok += 1
    return ok / tot if tot else 0.0


def test_e_uniforme_no_supera_azar():
    rng = random.Random(0)
    acts = list(Accion)
    seq = [rng.choice(acts) for _ in range(4000)]
    tr, te = seq[:2000], seq[2000:]
    limite = 1.0 / 7.0 + 0.05
    assert _media_acc1_repetir(te) <= limite
    assert _media_acc1_ngram(tr, te) <= limite
    assert _media_acc1_azar(te, seed=0) <= limite
    tr_eps = _eps_uniformes(11, 20, 100)
    te_eps = _eps_uniformes(12, 10, 100)
    m = Atisbo.entrenar(tr_eps, seed=0, epocas=2)
    assert _acc1_atisbo(m, te_eps) <= limite


def test_f_deterministas_y_validas():
    h = [Accion.ARRIBA, Accion.ABAJO]
    a1 = repetir_ultima(h)
    a2 = repetir_ultima(h)
    assert a1 == a2
    for a in a1:
        assert isinstance(a, Accion)
    ng1 = ngram3()
    ng1.ajustar([[Accion.ARRIBA, Accion.ABAJO, Accion.IZQ]])
    p1 = ng1.predecir(h)
    ng2 = ngram3()
    ng2.ajustar([[Accion.ARRIBA, Accion.ABAJO, Accion.IZQ]])
    p2 = ng2.predecir(h)
    assert p1 == p2
    for a in p1:
        assert isinstance(a, Accion)


def _corre_rapido(tmp_path):
    out = tmp_path / "E0.json"
    r = subprocess.run(
        [sys.executable, "-m", "atisbo.exp.e0", "--rapido", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    return json.loads(out.read_text(encoding="utf-8"))


def test_g_schema(tmp_path):
    datos = _corre_rapido(tmp_path)
    esquema = json.loads(Path("atisbo/exp/schema_results.json").read_text(encoding="utf-8"))
    validar(datos, esquema)


def test_h_reproducible_salvo_fecha(tmp_path):
    d1 = _corre_rapido(tmp_path)
    d2 = _corre_rapido(tmp_path)
    d1.pop("fecha_utc", None)
    d2.pop("fecha_utc", None)
    assert d1 == d2


def test_i_bootstrap():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    b1 = bootstrap(vals, n_remuestreos=1000, seed=0)
    b2 = bootstrap(vals, n_remuestreos=1000, seed=0)
    assert b1 == b2
    assert b1["ic95"][0] <= b1["media"] <= b1["ic95"][1]
