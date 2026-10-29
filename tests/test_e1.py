"""Anclas E1 (PLIEGO §5/§8/§9 variante LLM en cada paso). Sin red: simulada o replay."""
from __future__ import annotations

import json
from pathlib import Path

from atisbo.control import PredictorNulo
from atisbo.env.mundo import es_exito
from atisbo.env.rutina import elegir_accion
from atisbo.env.trazas import generar
from atisbo.exp.comun import validar
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import ClienteMeta
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada


class TransporteFalso:
    def __init__(self, cuerpos):
        self.cuerpos = list(cuerpos)
        self.llamadas = []

    def post(self, url, cabeceras, cuerpo):
        self.llamadas.append((url, cuerpo))
        idx = min(len(self.llamadas) - 1, len(self.cuerpos) - 1)
        return self.cuerpos[idx]


def _resp_ok(texto):
    body = json.dumps({"choices": [{"message": {"content": texto}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode("utf-8")
    return (200, {}, body)


def test_correr_episodio_nulo_simulada_todo_llm():
    mundo = Mundo(seed=0, p_perturb=0.0)
    pol = PoliticaSimulada(seed=0)
    registro = []
    res = correr_episodio(PredictorNulo(), pol, mundo, tau=0.5, epsilon=1e9, registro=registro)
    assert res["pasos"] > 0
    assert res["llamadas_llm"] == res["pasos"]
    assert len(registro) == res["pasos"]
    assert all(e["fuente"] == "llm" for e in registro)


def test_mundo_con_rutina_reproduce_generar():
    seed, n, p = 3, 3, 0.02
    ref = generar(seed, n, p)
    mundo = Mundo(seed=seed, p_perturb=p)
    assert len(ref) == n
    for ep_ref in ref:
        s = mundo.reiniciar()
        pasos = []
        for _ in range(160):
            if es_exito(s):
                break
            a = elegir_accion(s)
            pasos.append((s, a))
            s = mundo.avanzar(s, a)
        assert pasos == list(ep_ref)
        assert list(mundo.perturbaciones) == list(ep_ref.perturbaciones)
        assert mundo.exito == bool(ep_ref.exito)


def test_mundo_reproduce_generar_con_episodios_inacabados():
    seed, n, p = 1, 30, 0.05
    ref = generar(seed, n, p)
    assert any(not ep.exito for ep in ref)
    mundo = Mundo(seed=seed, p_perturb=p)
    assert len(ref) == n
    for ep_ref in ref:
        s = mundo.reiniciar()
        pasos = []
        for _ in range(160):
            if es_exito(s):
                break
            a = elegir_accion(s)
            pasos.append((s, a))
            s = mundo.avanzar(s, a)
        assert pasos == list(ep_ref)
        assert list(mundo.perturbaciones) == list(ep_ref.perturbaciones)
        assert mundo.exito == bool(ep_ref.exito)

def test_correr_episodio_para_al_truncarse(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "X")
    fake = TransporteFalso([_resp_ok('{"accion": "IZQ"}')] * 10)
    cli = ClienteMeta(modo="api", cache_dir=tmp_path, transporte=fake, dormir=lambda s: None)
    pol = PoliticaLLM(cliente=cli, max_llm_calls=2)
    mundo = Mundo(seed=0, p_perturb=0.0)
    registro = []
    res = correr_episodio(PredictorNulo(), pol, mundo, tau=0.5, epsilon=1e9, registro=registro)
    assert pol.truncada is True
    assert res["truncada"] is True
    assert res["pasos"] == 2
    assert res["llamadas_llm"] == 2
    assert len(registro) == 2
    assert len(fake.llamadas) == 2


def test_e1_simulada_esquema_valido(tmp_path):
    from atisbo.exp import e1
    out = tmp_path / "E1.json"
    r = e1.main(["--seed", "0", "--n", "2", "--p", "0.0", "--max-llm-calls", "50", "--cache", str(tmp_path / "cache"), "--out", str(out), "--modo", "simulada"])
    data = json.loads(Path(r).read_text(encoding="utf-8"))
    assert data["experimento"] == "E1"
    assert data["llm"]["modo"] == "simulada"
    assert data["llm"]["modelo"] == "muse-spark-1.3"
    assert "truncada" in data
    esquema = json.loads(Path("atisbo/exp/schema_results.json").read_text(encoding="utf-8"))
    validar(data, esquema)


def test_e1_modo_sale_de_env_si_no_modo(tmp_path, monkeypatch):
    from atisbo.exp import e1
    monkeypatch.setenv("ATISBO_LLM", "simulada")
    monkeypatch.delenv("META_API_KEY", raising=False)
    out = tmp_path / "E1.json"
    r = e1.main(["--seed", "0", "--n", "1", "--p", "0.0", "--max-llm-calls", "10", "--cache", str(tmp_path / "cache"), "--out", str(out)])
    data = json.loads(Path(r).read_text(encoding="utf-8"))
    assert data["llm"]["modo"] == "simulada"


def test_e1_replay_dos_primeros_episodios(tmp_path, monkeypatch):
    from atisbo.exp import e1
    monkeypatch.delenv("META_API_KEY", raising=False)
    fixtures = Path(__file__).parent / "fixtures" / "llm"
    assert len(list(fixtures.glob("*.json"))) == 128
    class TransporteProhibido:
        def post(self, *args, **kwargs):
            raise AssertionError("replay no debe tocar la red")
    out = tmp_path / "E1.json"
    r = e1.main(["--seed", "0", "--n", "2", "--p", "0.02", "--cache", str(fixtures), "--out", str(out), "--modo", "replay"], transporte=TransporteProhibido())
    data = json.loads(Path(r).read_text(encoding="utf-8"))
    esquema = json.loads(Path("atisbo/exp/schema_results.json").read_text(encoding="utf-8"))
    validar(data, esquema)
    assert data["llm"]["modo"] == "replay"
    assert data["truncada"] is False
    assert data["config"]["episodios_completados"] == 2
    assert data["llamadas_llm"] == 128
    assert len(list(fixtures.glob("*.json"))) == 128
