import numpy as np
from atisbo.env.trazas import generar
from atisbo.estado import Accion, vector
from atisbo.modelo import Atisbo
from atisbo.control import PredictorOraculo, PredictorNulo


def _ventana(ep, idx, k=8):
    n = len(ep)
    vecs = []
    for j in range(k):
        ii = max(0, idx - (k - 1 - j))
        ii = min(ii, n - 1)
        vecs.append(vector(ep[ii][0]))
    return np.stack(vecs).astype(np.float32)


def _acc1(modelo, episodios):
    tot = 0
    ok = 0
    for ep in episodios:
        if len(ep) < 4:
            continue
        for i in range(len(ep) - 3):
            v = _ventana(ep, i)
            pred = modelo.predecir(v)
            tot += 1
            if pred.acciones[0] == ep[i][1]:
                ok += 1
    return ok / tot if tot else 0.0


def test_a_entrenar_y_predecir():
    eps = generar(0, 20, 0.0)
    m = Atisbo.entrenar(eps, seed=0, epocas=2)
    assert m.n_parametros() <= 1_000_000
    ep = eps[0]
    v = _ventana(ep, 0)
    p = m.predecir(v)
    assert len(p.acciones) == 3
    for a in p.acciones:
        assert isinstance(a, Accion)
    assert p.estados.shape[1] == vector(ep[0][0]).shape[0]
    assert p.estados.shape[0] == 3
    assert 0.0 <= float(p.confianza) <= 1.0


def test_b_determinismo_y_guardar(tmp_path):
    eps = generar(0, 20, 0.0)
    m1 = Atisbo.entrenar(eps, seed=0, epocas=2)
    m2 = Atisbo.entrenar(eps, seed=0, epocas=2)
    d1 = m1.state_dict()
    d2 = m2.state_dict()
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        assert bool((d1[k] == d2[k]).all())
    v = _ventana(eps[0], 0)
    p1 = m1.predecir(v)
    ruta = tmp_path / "m.pt"
    m1.guardar(ruta)
    m3 = Atisbo.cargar(ruta)
    p3 = m3.predecir(v)
    assert tuple(p1.acciones) == tuple(p3.acciones)
    assert bool((p1.estados == p3.estados).all())
    assert float(p1.confianza) == float(p3.confianza)


def test_c_sanidad():
    import random
    eps = generar(1, 200, 0.0)
    rng = random.Random(1)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    ntr = int(0.8 * len(eps))
    tr = [eps[i] for i in idx[:ntr]]
    te = [eps[i] for i in idx[ntr:]]
    m = Atisbo.entrenar(tr, seed=1, epocas=10)
    assert _acc1(m, te) >= 0.75


def test_d_control():
    eps = generar(2, 5, 0.0)
    ora = PredictorOraculo()
    nulo = PredictorNulo()
    tot = 0
    ok = 0
    for ep in eps:
        if len(ep) < 4:
            continue
        for i in range(len(ep) - 3):
            v = _ventana(ep, i)
            p = ora.predecir(v)
            tot += 1
            if p.acciones[0] == ep[i][1]:
                ok += 1
            pn = nulo.predecir(v)
            assert float(pn.confianza) == 0.0
    assert tot > 0
    assert ok / tot == 1.0
