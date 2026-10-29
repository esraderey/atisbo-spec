"""Remates O3 parte 2: E2 y h1 (e1-e4). Sin red."""
from __future__ import annotations

import json
from pathlib import Path

import atisbo.exp.e2 as e2
from atisbo.exp.comun import validar


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema() -> dict:
    return json.loads((_repo_root() / "atisbo" / "exp" / "schema_results.json").read_text(encoding="utf-8"))


def _e1_path() -> Path:
    return _repo_root() / "results" / "E1.json"


def test_e1_decidir_h1_bordes():
    f = e2.decidir_h1
    ok, det = f(27.615, 1.0, 92.05, 1.0, False)
    assert ok is True
    assert det["ratio_llamadas"] == 27.615 / 92.05
    assert det["umbral_ratio"] == 0.3
    assert det["umbral_delta_pp"] == -5.0
    ok2, _ = f(10.0, 0.95, 100.0, 1.0, False)
    assert (0.95 - 1.0) * 100.0 != -5.0
    assert ok2 is True
    ok3, _ = f(31.0, 1.0, 100.0, 1.0, False)
    assert ok3 is False
    ok4, _ = f(10.0, 0.94, 100.0, 1.0, False)
    assert ok4 is False
    ok5, _ = f(27.615, 1.0, 92.05, 1.0, True)
    assert ok5 is False
    ok6, _ = f(0.0, 1.0, 0.0, 1.0, False)
    assert ok6 is False


def test_e2_procedencia_datos(tmp_path, monkeypatch):
    orig_gen = e2.generar_trazas
    orig_split = e2.split_episodios
    orig_cal = e2.calibrar_epsilon
    orig_mundo = e2.Mundo
    orig_correr = e2.correr_episodio
    import atisbo.modelo as modelo_mod

    orig_entrenar = modelo_mod.Atisbo.entrenar
    gen_ret = []
    gen_calls = []

    def spy_gen(seed, n, p):
        gen_calls.append((int(seed), int(n), float(p)))
        r = orig_gen(seed, n, p)
        gen_ret.append(r)
        return r

    split_calls = []
    split_ret = []

    def spy_split(eps, seed, *a, **k):
        tr, te = orig_split(eps, seed, *a, **k)
        split_calls.append((eps, int(seed)))
        split_ret.append((tr, te))
        return tr, te

    cal_calls = []

    def spy_cal(modelo, eps):
        cal_calls.append((modelo, eps))
        return orig_cal(modelo, eps)

    mundo_calls = []

    def spy_mundo(*a, **k):
        inst = orig_mundo(*a, **k)
        mundo_calls.append((a, k))
        return inst

    correr_calls = []

    def spy_correr(predictor, politica, mundo, tau=None, epsilon=None, registro=None, **kw):
        correr_calls.append({"epsilon": float(epsilon), "mundo": mundo, "tau": float(tau) if tau is not None else None})
        return orig_correr(predictor, politica, mundo, tau=tau, epsilon=epsilon, registro=registro, **kw)

    entrenar_calls = []

    def spy_entrenar(episodios, seed=None, epocas=None, **kw):
        entrenar_calls.append((episodios, int(seed) if seed is not None else None))
        return orig_entrenar(episodios, seed=seed, epocas=epocas, **kw)

    monkeypatch.setattr(e2, "generar_trazas", spy_gen)
    monkeypatch.setattr(e2, "split_episodios", spy_split)
    monkeypatch.setattr(e2, "calibrar_epsilon", spy_cal)
    monkeypatch.setattr(e2, "Mundo", spy_mundo)
    monkeypatch.setattr(e2, "correr_episodio", spy_correr)
    monkeypatch.setattr(modelo_mod.Atisbo, "entrenar", spy_entrenar)
    out = tmp_path / "E2.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e1", str(_e1_path())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    s = int(datos["seeds"][0])
    assert gen_calls == [(s, 20, 0.02), (s, 20, 0.0)]
    assert len(split_ret) == 2
    assert split_calls[0][0] is gen_ret[0]
    assert split_calls[1][0] is gen_ret[1]
    assert split_calls[0][1] == s
    assert split_calls[1][1] == s
    assert len(entrenar_calls) == 1
    assert entrenar_calls[0][0] is split_ret[0][0]
    assert len(cal_calls) == 1
    assert cal_calls[0][1] is split_ret[1][1]
    assert len(mundo_calls) == 1
    a, k = mundo_calls[0]
    semilla = int(k["seed"]) if "seed" in k else int(a[0])
    assert semilla == 1000 + s
    assert len(correr_calls) == int(datos["config"]["n_episodios_por_seed"])
    eps_cfg = float(datos["config"]["epsilon_por_seed"][0])
    for c in correr_calls:
        assert c["epsilon"] == eps_cfg


class _TransporteProhibido:
    def post(self, url, cabeceras, cuerpo):
        raise AssertionError("no debe tocar la red")


def test_e3_truncado_real(tmp_path):
    cache = tmp_path / "cache-vacia"
    cache.mkdir()
    out = tmp_path / "E2.json"
    e2.main(
        ["--rapido", "--modo", "replay", "--max-llm-calls", "0", "--cache", str(cache), "--out", str(out), "--e1", str(_e1_path())],
        transporte=_TransporteProhibido(),
    )
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert datos["truncada"] is True
    assert datos["h1"] is False
    assert list(cache.rglob("*.json")) == []
    validar(datos, _schema())


def test_e4_metricas_nuevas(tmp_path):
    out = tmp_path / "E2.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e1", str(_e1_path())])
    datos = json.loads(out.read_text(encoding="utf-8"))
    for clave in ("pasos", "malas_aceptadas", "llamadas_por_sorpresa"):
        assert clave in datos["metricas"]
        b = datos["metricas"][clave]
        assert "media" in b and "ic95" in b and "por_seed" in b
        assert isinstance(b["por_seed"], list) and len(b["por_seed"]) >= 1
    validar(datos, _schema())


def test_r12_e1_truncada_bloquea_h1(tmp_path):
    base = json.loads(_e1_path().read_text(encoding="utf-8"))
    base["truncada"] = True
    base["metricas"]["llamadas_llm"]["media"] = 1e6
    base["exito_media"] = 0.0
    e1_falso = tmp_path / "E1-falso.json"
    e1_falso.write_text(json.dumps(base), encoding="utf-8")
    out = tmp_path / "E2.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out), "--e1", str(e1_falso)])
    datos = json.loads(out.read_text(encoding="utf-8"))
    assert datos["h1"] is False
    base["truncada"] = False
    e1_falso.write_text(json.dumps(base), encoding="utf-8")
    out2 = tmp_path / "E2b.json"
    e2.main(["--rapido", "--modo", "simulada", "--out", str(out2), "--e1", str(e1_falso)])
    datos2 = json.loads(out2.read_text(encoding="utf-8"))
    assert datos2["h1"] is True
