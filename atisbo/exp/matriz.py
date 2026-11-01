"""Matriz de ablaciones y robustez (PLIEGO §9 E4 y E5). Reusa lazo y e2."""
from __future__ import annotations

from dataclasses import dataclass

from atisbo.exp.comun import bootstrap, generar_trazas, split_episodios
from atisbo.exp.e2 import PRECIO_IN_MUSD, PRECIO_OUT_MUSD, _media, semilla_mundo
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import MODELO, CacheMiss, ClienteMeta
from atisbo.modelo import Atisbo
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada
from atisbo.sorpresa import calibrar_epsilon


@dataclass(frozen=True)
class Celda:
    nombre: str
    p: float = 0.02
    tau: float = 0.5
    sin_sorpresa: bool = False
    horizonte: int = 3


def correr_matriz(celdas, seeds, n, n_trazas, epocas, modo, cache, max_llm_calls, transporte=None) -> dict:
    celdas = list(celdas)
    seeds = [int(s) for s in seeds]
    n = int(n)
    n_trazas = int(n_trazas)
    epocas = int(epocas)
    max_llm_calls = int(max_llm_calls)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds duplicadas")
    if modo not in ("api", "replay", "simulada"):
        raise ValueError("modo api|replay|simulada")
    if modo == "simulada":
        cliente = None
        compartida = None
    else:
        cliente = ClienteMeta(modo=modo, cache_dir=cache, transporte=transporte)
        compartida = PoliticaLLM(cliente=cliente, max_llm_calls=max_llm_calls)
    modelos: dict = {}
    epsilons: dict = {}
    for s in seeds:
        eps_tr = generar_trazas(int(s), n_trazas, 0.02)
        tr, _ = split_episodios(eps_tr, int(s))
        modelos[int(s)] = Atisbo.entrenar(tr, seed=int(s), epocas=epocas)
        eps_cal = generar_trazas(int(s), n_trazas, 0.0)
        _, te0 = split_episodios(eps_cal, int(s))
        epsilons[int(s)] = float(calibrar_epsilon(modelos[int(s)], te0))
    epsilon_lista = [float(epsilons[int(s)]) for s in seeds]
    por_celda: dict = {}
    for celda in celdas:
        por_celda[celda.nombre] = {
            "celda": celda,
            "llamadas": [],
            "exito": [],
            "pasos": [],
            "racha": [],
            "tasa": [],
            "xsorpresa": [],
            "malas": [],
            "por_seed": {int(s): {"llamadas": [], "exito": [], "pasos": [], "racha": [], "tasa": [], "xsorpresa": [], "malas": []} for s in seeds},
            "cache_miss": 0,
            "n_ok": 0,
        }
    total_llamadas = 0
    total_peticiones = 0
    total_invalidas = 0
    truncada = False
    pasos_parcial = 0
    for s in seeds:
        if truncada:
            break
        modelo = modelos[int(s)]
        for celda in celdas:
            if truncada:
                break
            acc = por_celda[celda.nombre]
            if modo == "simulada":
                politica_cs = PoliticaSimulada(seed=int(s))
            else:
                politica_cs = compartida
            mundo = Mundo(seed=semilla_mundo(int(s)), p_perturb=float(celda.p))
            for _ in range(n):
                if truncada:
                    break
                epsilon = float("inf") if bool(celda.sin_sorpresa) else float(epsilons[int(s)])
                registro: list = []
                previo = (cliente.tokens_entrada, cliente.tokens_salida, compartida.peticiones, compartida.llamadas_llm, compartida.llamadas_red, compartida.llm_invalidas) if cliente is not None else None
                try:
                    res = correr_episodio(modelo, politica_cs, mundo, tau=float(celda.tau), epsilon=float(epsilon), registro=registro, horizonte=int(celda.horizonte))
                except CacheMiss:
                    cliente.tokens_entrada, cliente.tokens_salida, compartida.peticiones, compartida.llamadas_llm, compartida.llamadas_red, compartida.llm_invalidas = previo
                    acc["cache_miss"] += 1
                    continue
                if bool(res["truncada"]):
                    truncada = True
                    pasos_parcial = int(res["pasos"])
                    total_llamadas += int(res["llamadas_llm"])
                    total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
                    total_invalidas += int(res["llm_invalidas"])
                    break
                acc["llamadas"].append(float(res["llamadas_llm"]))
                acc["exito"].append(1.0 if res["exito"] else 0.0)
                acc["pasos"].append(float(res["pasos"]))
                acc["racha"].append(float(res["racha_media"]))
                acc["tasa"].append(float(res["tasa_malas_aceptadas"]))
                acc["xsorpresa"].append(float(res["llamadas_por_sorpresa"]))
                acc["malas"].append(float(res["malas_aceptadas"]))
                ps = acc["por_seed"][int(s)]
                ps["llamadas"].append(float(res["llamadas_llm"]))
                ps["exito"].append(1.0 if res["exito"] else 0.0)
                ps["pasos"].append(float(res["pasos"]))
                ps["racha"].append(float(res["racha_media"]))
                ps["tasa"].append(float(res["tasa_malas_aceptadas"]))
                ps["xsorpresa"].append(float(res["llamadas_por_sorpresa"]))
                ps["malas"].append(float(res["malas_aceptadas"]))
                total_llamadas += int(res["llamadas_llm"])
                total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
                total_invalidas += int(res["llm_invalidas"])
                acc["n_ok"] += 1
    celdas_json: list = []
    metricas_planas: dict = {}
    for celda in celdas:
        acc = por_celda[celda.nombre]
        n_ok = int(acc["n_ok"])
        miss = int(acc["cache_miss"])
        evaluadas = [int(s) for s in seeds if acc["por_seed"][int(s)]["llamadas"]]
        if n_ok:
            mets: dict = {}
            for clave, vals, sk in (
                ("llamadas_llm", acc["llamadas"], "llamadas"),
                ("exito", acc["exito"], "exito"),
                ("pasos", acc["pasos"], "pasos"),
                ("racha_media", acc["racha"], "racha"),
                ("tasa_malas_aceptadas", acc["tasa"], "tasa"),
                ("llamadas_por_sorpresa", acc["xsorpresa"], "xsorpresa"),
                ("malas_aceptadas", acc["malas"], "malas"),
            ):
                b = bootstrap(vals, n_remuestreos=1000, seed=0)
                b["por_seed"] = [_media(acc["por_seed"][int(s)][sk]) for s in evaluadas]
                mets[clave] = b
                if not (n_ok > 0 and miss > 0):
                    metricas_planas[f"{clave}@{celda.nombre}"] = dict(b)
        else:
            mets = {}
        entrada: dict = {
            "nombre": str(celda.nombre),
            "config": {
                "p": float(celda.p),
                "tau": float(celda.tau),
                "epsilon_por_seed": [float(v) for v in epsilon_lista],
                "sin_sorpresa": bool(celda.sin_sorpresa),
                "horizonte": int(celda.horizonte),
            },
            "n_episodios": int(n_ok),
            "cache_miss": int(miss),
            "seeds_evaluadas": [int(s) for s in evaluadas],
            "metricas": mets,
        }
        if n_ok == 0 and miss > 0:
            entrada["omitida"] = "sin caché"
        if n_ok > 0 and miss > 0:
            entrada["parcial"] = "solo episodios en caché; no comparable con otras celdas"
        celdas_json.append(entrada)
    if cliente is not None:
        tok_in = int(cliente.tokens_entrada)
        tok_out = int(cliente.tokens_salida)
        red_total = int(getattr(compartida, "llamadas_red", 0))
    else:
        tok_in = 0
        tok_out = 0
        red_total = 0
    coste = (tok_in * PRECIO_IN_MUSD + tok_out * PRECIO_OUT_MUSD) / 1e6
    total_miss = int(sum(por_celda[c.nombre]["cache_miss"] for c in celdas))
    config = {
        "seeds": [int(s) for s in seeds],
        "n_episodios_por_seed": int(n),
        "n_trazas": int(n_trazas),
        "epocas": int(epocas),
        "modo": str(modo),
        "max_llm_calls": int(max_llm_calls),
        "cache": str(cache),
        "semillas_mundo": [int(semilla_mundo(int(s))) for s in seeds],
        "modelo": MODELO,
    }
    return {
        "config": config,
        "celdas": celdas_json,
        "metricas": metricas_planas,
        "llm": {
            "modo": str(modo),
            "modelo": MODELO,
            "llamadas": int(total_llamadas),
            "peticiones": int(total_peticiones),
            "llamadas_red": int(red_total),
            "tokens_entrada": int(tok_in),
            "tokens_salida": int(tok_out),
            "coste_estimado_usd": float(coste),
            "cache_miss": int(total_miss),
            "llm_invalidas": int(total_invalidas),
        },
        "truncada": bool(truncada),
        "pasos_parcial": int(pasos_parcial),
    }
