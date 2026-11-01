"""Experimento E4 (PLIEGO §9): ablaciones sobre E2. Sin h1."""
from __future__ import annotations

import argparse
import json
import os

from atisbo.exp.comun import commit_hash, escribir_json, fecha_utc
from atisbo.exp.matriz import Celda, correr_matriz


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="E4: ablaciones.")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-trazas", type=int, default=200)
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--max-llm-calls", type=int, default=1500)
    ap.add_argument("--cache", type=str, default="cache/llm")
    ap.add_argument("--out", type=str, default="results/E4.json")
    ap.add_argument("--e3", type=str, default="results/E3.json")
    ap.add_argument("--e2-sam", type=str, default="results/E2_sam.json")
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
    return seeds, n, n_trazas, epocas, int(a.max_llm_calls), str(a.cache), str(a.out), str(a.e3), str(a.e2_sam), a.modo, rapido


def _referencias(e3_path: str, e2sam_path: str) -> dict:
    with open(e3_path, encoding="utf-8") as f:
        e3d = json.load(f)
    conforme = []
    for entrada in e3d["por_alpha"]:
        a = float(entrada["alpha"])
        item = {
            "alpha": float(entrada["alpha"]),
            "llamadas_llm": float(entrada["llamadas_llm"]) if entrada["llamadas_llm"] is not None else None,
            "exito": float(entrada["exito"]) if entrada["exito"] is not None else None,
            "riesgo_empirico": float(entrada["riesgo_empirico"]) if entrada["riesgo_empirico"] is not None else None,
            "commit": e3d["commit"],
            "modo": e3d["llm"]["modo"],
            "seeds_evaluadas": [int(x) for x in e3d["config"]["seeds_evaluadas"]],
            "n_episodios": int(entrada["n_episodios"]),
        }
        item["llamadas_llm_ic95"] = [float(v) for v in e3d["metricas"][f"llamadas_llm@{a}"]["ic95"]]
        item["exito_ic95"] = [float(v) for v in e3d["metricas"][f"exito@{a}"]["ic95"]]
        conforme.append(item)
    try:
        with open(e2sam_path, encoding="utf-8") as f2:
            e2s = json.load(f2)
    except OSError:
        return {"conforme": conforme, "e_sam": {"omitida": "sin O5"}}
    e_sam = {
        "llamadas_llm": float(e2s["metricas"]["llamadas_llm"]["media"]),
        "exito": float(e2s["metricas"]["exito"]["media"]),
        "coincidencia_todo": float(e2s["metricas"]["coincidencia_todo"]["media"]),
        "comparacion_e2": dict(e2s["comparacion_e2"]),
        "commit": e2s["commit"],
        "modo": e2s["llm"]["modo"],
        "seeds_evaluadas": [int(x) for x in e2s["config"]["seeds_evaluadas"]],
        "n_episodios": int(e2s["config"]["episodios_completados"]),
        "llamadas_llm_ic95": [float(v) for v in e2s["metricas"]["llamadas_llm"]["ic95"]],
        "exito_ic95": [float(v) for v in e2s["metricas"]["exito"]["ic95"]],
    }
    return {"conforme": conforme, "e_sam": e_sam}


def main(argv=None, transporte=None) -> str:
    seeds, n, n_trazas, epocas, max_llm_calls, cache, out, e3_path, e2sam_path, modo_arg, rapido = _parse_args(argv)
    modo = modo_arg if modo_arg is not None else os.environ.get("ATISBO_LLM", "simulada")
    if modo not in ("api", "replay", "simulada"):
        raise ValueError("modo api|replay|simulada")
    referencias = _referencias(e3_path, e2sam_path)
    celdas = [
        Celda(nombre="base", p=0.02, tau=0.5, sin_sorpresa=False, horizonte=3),
        Celda(nombre="sin_sorpresa", p=0.02, tau=0.5, sin_sorpresa=True, horizonte=3),
        Celda(nombre="horizonte_1", p=0.02, tau=0.5, sin_sorpresa=False, horizonte=1),
    ]
    res = correr_matriz(celdas, seeds, n, n_trazas, epocas, modo, cache, max_llm_calls, transporte=transporte)
    referencias["comparable"] = bool(not rapido and modo != "simulada")
    notas = [
        "E4 ablaciones sobre E2: base es E2 con tau fija 0.5, sorpresa y horizonte 3.",
        "El coste estimado cuenta tambien lo servido desde la cache.",
        "cache_miss cuenta episodios omitidos en replay por falta de cache; no se improvisa politica.",
        "bootstrap 1000 remuestreos semilla 0 sobre medias por episodio y por celda.",
        "los episodios truncados no entran en las metricas pero si en llamadas, tokens y coste.",
        "en sin_sorpresa el epsilon efectivo es inf aunque epsilon_por_seed muestre el calibrado.",
        "llamadas_por_sorpresa vale 0 por construccion en sin_sorpresa y en horizonte_1.",
        "las celdas comparten estados iniciales pero no van pareadas, porque las perturbaciones divergen tras el primer episodio de distinta longitud.",
    ]
    if bool(res["truncada"]):
        notas.append("corrida truncada por tope LLM: el parcial queda fuera de las metricas y en pasos_parcial.")
    config = dict(res["config"])
    config["rapido"] = bool(rapido)
    datos = {
        "experimento": "E4",
        "fecha_utc": fecha_utc(),
        "commit": commit_hash(),
        "seeds": [int(s) for s in seeds],
        "config": config,
        "metricas": dict(res["metricas"]),
        "celdas": list(res["celdas"]),
        "llm": dict(res["llm"]),
        "truncada": bool(res["truncada"]),
        "pasos_parcial": int(res["pasos_parcial"]),
        "referencias": referencias,
        "notas": notas,
    }
    escribir_json(out, datos)
    llm = res["llm"]
    print(f"[E4] modo={modo} llamadas={llm['llamadas']} peticiones={llm['peticiones']} llamadas_red={llm['llamadas_red']} cache_miss={llm['cache_miss']} truncada={res['truncada']}")
    return str(out)


if __name__ == "__main__":
    main()
