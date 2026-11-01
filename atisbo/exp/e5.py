"""Experimento E5 (PLIEGO §9): robustez a perturbaciones. Sin h1."""
from __future__ import annotations

import argparse
import os

from atisbo.exp.comun import commit_hash, escribir_json, fecha_utc
from atisbo.exp.matriz import Celda, correr_matriz


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="E5: robustez.")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-trazas", type=int, default=200)
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--max-llm-calls", type=int, default=1500)
    ap.add_argument("--cache", type=str, default="cache/llm")
    ap.add_argument("--out", type=str, default="results/E5.json")
    ap.add_argument("--modo", type=str, default=None)
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds duplicadas")
    n = int(a.n)
    n_trazas = int(a.n_trazas)
    epocas = int(a.epocas)
    rapido = bool(a.rapido)
    if rapido:
        seeds = [seeds[0] if seeds else 0]
        n_trazas = 20
        epocas = 2
        n = 2
    return seeds, n, n_trazas, epocas, int(a.max_llm_calls), str(a.cache), str(a.out), a.modo, rapido


def main(argv=None, transporte=None) -> str:
    seeds, n, n_trazas, epocas, max_llm_calls, cache, out, modo_arg, rapido = _parse_args(argv)
    modo = modo_arg if modo_arg is not None else os.environ.get("ATISBO_LLM", "simulada")
    if modo not in ("api", "replay", "simulada"):
        raise ValueError("modo api|replay|simulada")
    celdas = [
        Celda(nombre="p0", p=0.0, tau=0.5, sin_sorpresa=False, horizonte=3),
        Celda(nombre="p002", p=0.02, tau=0.5, sin_sorpresa=False, horizonte=3),
        Celda(nombre="p005", p=0.05, tau=0.5, sin_sorpresa=False, horizonte=3),
    ]
    res = correr_matriz(celdas, seeds, n, n_trazas, epocas, modo, cache, max_llm_calls, transporte=transporte)
    notas = [
        "E5 robustez: p0, p002 y p005 con lo demas como base E2.",
        "El coste estimado cuenta tambien lo servido desde la cache.",
        "cache_miss cuenta episodios omitidos en replay por falta de cache; no se improvisa politica.",
        "bootstrap 1000 remuestreos semilla 0 sobre medias por episodio y por celda.",
        "los episodios truncados no entran en las metricas pero si en llamadas, tokens y coste.",
        "las celdas comparten estados iniciales pero no van pareadas, porque las perturbaciones divergen tras el primer episodio de distinta longitud.",
    ]
    if bool(res["truncada"]):
        notas.append("corrida truncada por tope LLM: el parcial queda fuera de las metricas y en pasos_parcial.")
    config = dict(res["config"])
    config["rapido"] = bool(rapido)
    datos = {
        "experimento": "E5",
        "fecha_utc": fecha_utc(),
        "commit": commit_hash(),
        "seeds": [int(s) for s in seeds],
        "config": config,
        "metricas": dict(res["metricas"]),
        "celdas": list(res["celdas"]),
        "llm": dict(res["llm"]),
        "truncada": bool(res["truncada"]),
        "pasos_parcial": int(res["pasos_parcial"]),
        "notas": notas,
    }
    escribir_json(out, datos)
    llm = res["llm"]
    print(f"[E5] modo={modo} llamadas={llm['llamadas']} peticiones={llm['peticiones']} llamadas_red={llm['llamadas_red']} cache_miss={llm['cache_miss']} truncada={res['truncada']}")
    return str(out)


if __name__ == "__main__":
    main()
