"""Experimento E1 (PLIEGO 9): baseline caro con LLM en cada paso."""
from __future__ import annotations

import argparse
import os

import numpy as np

from atisbo.control import PredictorNulo
from atisbo.exp.comun import bootstrap, commit_hash, escribir_json, fecha_utc
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import MODELO, ClienteMeta
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada

TAU = 0.5
EPSILON = 1e9
PRECIO_IN_MUSD = 1.25
PRECIO_OUT_MUSD = 4.25


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="E1: baseline caro (LLM en cada paso).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--max-llm-calls", type=int, default=1500)
    ap.add_argument("--cache", type=str, default="cache/llm")
    ap.add_argument("--out", type=str, default="results/E1.json")
    ap.add_argument("--modo", type=str, default=None)
    return ap.parse_args(argv)


def _percentil(xs, q: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def main(argv=None, transporte=None) -> str:
    a = _parse_args(argv)
    modo = a.modo if a.modo is not None else os.environ.get("ATISBO_LLM", "simulada")
    if modo not in ("api", "replay", "simulada"):
        raise ValueError("modo api|replay|simulada")
    if modo == "simulada":
        politica = PoliticaSimulada(seed=int(a.seed))
        cliente = None
    else:
        cliente = ClienteMeta(modo=modo, cache_dir=a.cache, transporte=transporte)
        politica = PoliticaLLM(cliente=cliente, max_llm_calls=int(a.max_llm_calls))
    mundo = Mundo(seed=int(a.seed), p_perturb=float(a.p))
    predictor = PredictorNulo()
    por_llamadas = []
    por_exito = []
    por_lat_media = []
    todas_lat = []
    total_llamadas = 0
    total_peticiones = 0
    total_invalidas = 0
    truncada = False
    pasos_parcial = 0
    n_ok = 0
    for _ in range(int(a.n)):
        registro: list = []
        res = correr_episodio(predictor, politica, mundo, tau=TAU, epsilon=EPSILON, registro=registro)
        if bool(res["truncada"]):
            truncada = True
            pasos_parcial = int(res["pasos"])
            total_llamadas += int(res["llamadas_llm"])
            total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
            total_invalidas += int(res["llm_invalidas"])
            break
        por_llamadas.append(float(res["llamadas_llm"]))
        por_exito.append(1.0 if res["exito"] else 0.0)
        lats = [float(x) for x in res["latencias_ms"]]
        todas_lat.extend(lats)
        por_lat_media.append(float(sum(lats) / len(lats)) if lats else 0.0)
        total_llamadas += int(res["llamadas_llm"])
        total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
        total_invalidas += int(res["llm_invalidas"])
        n_ok += 1
    if n_ok:
        metricas = {
            "llamadas_llm": _con_por_seed(bootstrap(por_llamadas, n_remuestreos=1000, seed=0), por_llamadas),
            "exito": _con_por_seed(bootstrap(por_exito, n_remuestreos=1000, seed=0), por_exito),
            "latencia_ms": _con_por_seed(bootstrap(por_lat_media, n_remuestreos=1000, seed=0), por_lat_media),
        }
    else:
        metricas = {}
    if cliente is not None:
        tok_in = int(cliente.tokens_entrada)
        tok_out = int(cliente.tokens_salida)
        red_total = int(getattr(politica, "llamadas_red", 0))
    else:
        tok_in = 0
        tok_out = 0
        red_total = 0
    coste = (tok_in * PRECIO_IN_MUSD + tok_out * PRECIO_OUT_MUSD) / 1e6
    datos = {
        "experimento": "E1",
        "fecha_utc": fecha_utc(),
        "commit": commit_hash(),
        "seeds": [int(a.seed)],
        "config": {"seed": int(a.seed), "n_episodios": int(a.n), "episodios_completados": int(n_ok), "p_perturb": float(a.p), "max_llm_calls": int(a.max_llm_calls), "tau": float(TAU), "epsilon": float(EPSILON), "modelo": MODELO},
        "metricas": metricas,
        "llm": {"modo": modo, "modelo": MODELO, "llamadas": int(total_llamadas), "peticiones": int(total_peticiones), "tokens_entrada": int(tok_in), "tokens_salida": int(tok_out), "coste_estimado_usd": float(coste)},
        "llamadas_llm": int(total_llamadas),
        "latencia_p50_ms": _percentil(todas_lat, 50),
        "latencia_p95_ms": _percentil(todas_lat, 95),
        "exito_media": float(sum(por_exito) / len(por_exito)) if por_exito else 0.0,
        "llm_invalidas": int(total_invalidas),
        "coste_estimado_usd": float(coste),
        "truncada": bool(truncada),
        "pasos_parcial": int(pasos_parcial),
        "notas": ["E1 baseline caro: PredictorNulo, LLM en cada paso.", "Coste estimado con precios muse-spark-1.3: https://dev.meta.ai/docs/pricing-rate-limits", "bootstrap 1000 remuestreos semilla 0 sobre medias por episodio."],
    }
    escribir_json(a.out, datos)
    print(f"[E1] modo={modo} llamadas={total_llamadas} peticiones={total_peticiones} llamadas_red={red_total} truncada={truncada}")
    return str(a.out)


def _con_por_seed(b: dict, por_episodio) -> dict:
    b = dict(b)
    b["por_seed"] = [float(sum(por_episodio) / len(por_episodio))] if por_episodio else [0.0]
    return b


if __name__ == "__main__":
    main()
