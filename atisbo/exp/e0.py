"""Experimento E0 (PLIEGO §9): Atisbo vs baselines en acierto@1/@3."""

from __future__ import annotations

import argparse
import time

from atisbo.baselines import azar, ngram3, repetir_ultima
from atisbo.exp.comun import (
    bootstrap,
    commit_hash,
    escribir_json,
    fecha_utc,
    generar_trazas,
    split_episodios,
    ventana,
)
from atisbo.metrics import acierto_at_1, acierto_at_3
from atisbo.modelo import Atisbo

METODOS = ("atisbo", "repetir_ultima", "ngram3", "azar")
BASELINES_GO = ("repetir_ultima", "ngram3", "azar")
LIMITE_S = 300


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="E0: estructura predecible (§9).")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--p", dest="ps", action="append", type=float, default=None)
    ap.add_argument("--n-episodios", type=int, default=200)
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--out", type=str, default="results/E0.json")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    ps = a.ps if a.ps is not None else [0.0, 0.02]
    if a.rapido:
        seeds = [seeds[0] if seeds else 0]
        ps = [0.0]
        return seeds, ps, 20, 2, a.out, True
    return seeds, ps, int(a.n_episodios), int(a.epocas), a.out, False


def _eval_episodio(ep, modelo, ng, az):
    acc = {m: {"acc1": [], "acc3": []} for m in METODOS}
    n = len(ep)
    if n < 4:
        return None
    for t in range(n - 3):
        verdad = [ep[t][1], ep[t + 1][1], ep[t + 2][1]]
        pa = modelo.predecir(ventana(ep, t)).acciones
        acc["atisbo"]["acc1"].append(acierto_at_1(pa, verdad))
        acc["atisbo"]["acc3"].append(acierto_at_3(pa, verdad))
        hist = [a for _, a in ep[:t]]
        for nombre, pred in (
            ("repetir_ultima", repetir_ultima(hist)),
            ("ngram3", ng.predecir(hist)),
            ("azar", az.predecir(hist)),
        ):
            acc[nombre]["acc1"].append(acierto_at_1(pred, verdad))
            acc[nombre]["acc3"].append(acierto_at_3(pred, verdad))
    return {m: {"acc1": sum(v["acc1"]) / len(v["acc1"]), "acc3": sum(v["acc3"]) / len(v["acc3"])} for m, v in acc.items()}


def promedio_por_seed(valores) -> float:
    """Media por semilla; error si no hay episodios evaluables."""
    vals = list(valores)
    if not vals:
        raise ValueError("semilla sin episodios evaluables")
    return float(sum(vals) / len(vals))


def decidir_veredicto(metricas: dict, ps) -> str:
    """Regla go/no-go pura: go si atisbo@3 supera el mejor sup IC95 en todo p."""
    go_ps = []
    for p in ps:
        mejor_sup = max(
            metricas[f"{m}/acierto@3/p={float(p)}"]["ic95"][1] for m in BASELINES_GO
        )
        go_ps.append(bool(metricas[f"atisbo/acierto@3/p={float(p)}"]["media"] > mejor_sup))
    return "go" if go_ps and all(go_ps) else "no-go"


def main(argv=None) -> str:
    seeds, ps, n_eps, epocas, out, rapido = _parse_args(argv)
    per_ep = {p: {m: {"acc1": [], "acc3": []} for m in METODOS} for p in ps}
    per_seed = {p: {m: {"acc1": {s: [] for s in seeds}, "acc3": {s: [] for s in seeds}} for m in METODOS} for p in ps}
    tiempos_min = []
    tiempos_ok = True
    n_params = None
    for s in seeds:
        for p in ps:
            eps = generar_trazas(s, n_eps, p)
            tr, te = split_episodios(eps, s)
            t0 = time.perf_counter()
            modelo = Atisbo.entrenar(tr, seed=s, epocas=epocas)
            dt = time.perf_counter() - t0
            if dt > LIMITE_S:
                tiempos_ok = False
            tiempos_min.append(int(dt // 60))
            if n_params is None:
                n_params = int(modelo.n_parametros())
            ng = ngram3().ajustar([[a for _, a in ep] for ep in tr])
            az = azar(seed=0)
            for ep in te:
                r = _eval_episodio(ep, modelo, ng, az)
                if r is None:
                    continue
                for m in METODOS:
                    per_ep[p][m]["acc1"].append(r[m]["acc1"])
                    per_ep[p][m]["acc3"].append(r[m]["acc3"])
                    per_seed[p][m]["acc1"][s].append(r[m]["acc1"])
                    per_seed[p][m]["acc3"][s].append(r[m]["acc3"])
    metricas = {}
    for p in ps:
        for m in METODOS:
            for met in ("acc1", "acc3"):
                vals = per_ep[p][m][met]
                b = bootstrap(vals, n_remuestreos=1000, seed=0)
                ps_means = []
                for s in seeds:
                    v = per_seed[p][m][met][s]
                    ps_means.append(promedio_por_seed(v))
                b["por_seed"] = ps_means
                clave = f"{m}/acierto@{'1' if met == 'acc1' else '3'}/p={float(p)}"
                metricas[clave] = b
    veredicto = decidir_veredicto(metricas, ps)
    datos = {
        "experimento": "E0",
        "fecha_utc": fecha_utc(),
        "commit": commit_hash(),
        "seeds": list(seeds),
        "config": {
            "p_perturb": [float(p) for p in ps],
            "n_episodios": int(n_eps),
            "epocas": int(epocas),
            "k": 8,
            "limite_entrenamiento_s": LIMITE_S,
            "rapido": bool(rapido),
        },
        "metricas": metricas,
        "entrenamiento": {
            "n_parametros": int(n_params or 0),
            "limite_s": LIMITE_S,
            "cumple_limite": bool(tiempos_ok),
            "tiempo_min_por_semilla": [int(x) for x in tiempos_min],
            "tiempo_max_min": int(max(tiempos_min)) if tiempos_min else 0,
        },
        "veredicto": veredicto,
        "notas": [
            "go si media atisbo@3 > mejor sup IC95 baseline en todo p; si no, no-go (Plan B).",
            "tiempos en minutos enteros para JSON reproducible (§8); comprobacion <300s interna.",
            "bootstrap 1000 remuestreos semilla 0 sobre medias por episodio.",
        ],
    }
    escribir_json(out, datos)
    return out


if __name__ == "__main__":
    main()
