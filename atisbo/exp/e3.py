"""Experimento E3 (PLIEGO 7 y 9): calibracion conforme + lazo por alfa."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from atisbo.conformal import lambda_hat, riesgo, tau_desde_lambda
from atisbo.exp.comun import bootstrap, commit_hash, escribir_json, fecha_utc, generar_trazas, split_episodios, ventana
from atisbo.exp.e2 import _Cronometro, decidir_h1, semilla_mundo
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import MODELO, ClienteMeta
from atisbo.modelo import Atisbo
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada
from atisbo.sorpresa import calibrar_epsilon

PRECIO_IN_MUSD = 1.25
PRECIO_OUT_MUSD = 4.25

ALPHA_H1 = 0.05


def entrada_h1(por_alpha):
    for e in por_alpha:
        if abs(float(e["alpha"]) - float(ALPHA_H1)) < 1e-12:
            return e
    return None


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="E3: tau conforme por alfa.")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--alphas", type=str, default="0.01,0.05,0.10")
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-trazas", type=int, default=200)
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--n-cal", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--max-llm-calls", type=int, default=1500)
    ap.add_argument("--cache", type=str, default="cache/llm")
    ap.add_argument("--out", type=str, default="results/E3.json")
    ap.add_argument("--e1", type=str, default="results/E1.json")
    ap.add_argument("--modo", type=str, default=None)
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    alphas = [float(x) for x in a.alphas.split(",") if x.strip() != ""]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds duplicadas")
    if len(set(alphas)) != len(alphas):
        raise ValueError("alphas duplicadas")
    n = int(a.n)
    n_trazas = int(a.n_trazas)
    epocas = int(a.epocas)
    n_cal = int(a.n_cal)
    n_test = int(a.n_test)
    rapido = bool(a.rapido)
    if rapido:
        seeds = [seeds[0] if seeds else 0]
        n_trazas = 20
        epocas = 2
        n_cal = 4
        n_test = 4
        n = 2
    return seeds, alphas, float(a.p), n, n_trazas, epocas, n_cal, n_test, int(a.max_llm_calls), str(a.cache), str(a.out), str(a.e1), a.modo, rapido


def _media(vals) -> float:
    vals = list(vals)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _cociente(pares) -> float:
    num = sum(int(p[0]) for p in pares)
    den = sum(int(p[1]) for p in pares)
    return float(num) / float(den) if den else 0.0


def _bootstrap_cociente(pares, n_remuestreos: int = 1000, seed: int = 0) -> dict:
    n = len(pares)
    if n == 0:
        raise ValueError("bootstrap sin valores")
    media = _cociente(pares)
    rng = np.random.RandomState(seed)
    nums = np.asarray([float(p[0]) for p in pares], dtype=np.float64)
    dens = np.asarray([float(p[1]) for p in pares], dtype=np.float64)
    medias = np.empty(int(n_remuestreos), dtype=np.float64)
    for b in range(int(n_remuestreos)):
        idx = rng.randint(0, n, size=n)
        d = float(dens[idx].sum())
        medias[b] = float(nums[idx].sum() / d) if d else 0.0
    lo = float(np.percentile(medias, 2.5))
    hi = float(np.percentile(medias, 97.5))
    return {"media": float(media), "ic95": [lo, hi], "n": int(n), "n_remuestreos": int(n_remuestreos)}


def _scores_malas(modelo, episodios):
    s_list: list = []
    m_list: list = []
    por_ep: list = []
    for ep in episodios:
        ss: list = []
        mm: list = []
        for t in range(len(ep)):
            pred = modelo.predecir(ventana(ep, t))
            s_list.append(float(1.0 - float(pred.confianza)))
            mala = 1 if pred.acciones[0] != ep[t][1] else 0
            m_list.append(mala)
            ss.append(float(1.0 - float(pred.confianza)))
            mm.append(mala)
        por_ep.append((ss, mm))
    return np.asarray(s_list, dtype=np.float64), np.asarray(m_list, dtype=np.int64), por_ep


def main(argv=None, transporte=None) -> str:
    seeds, alphas, p, n, n_trazas, epocas, n_cal, n_test, max_llm_calls, cache, out, e1_path, modo_arg, rapido = _parse_args(argv)
    modo = modo_arg if modo_arg is not None else os.environ.get("ATISBO_LLM", "simulada")
    if modo not in ("api", "replay", "simulada"):
        raise ValueError("modo api|replay|simulada")
    if modo == "simulada":
        cliente = None
        politica_compartida = None
    else:
        cliente = ClienteMeta(modo=modo, cache_dir=cache, transporte=transporte)
        politica_compartida = PoliticaLLM(cliente=cliente, max_llm_calls=int(max_llm_calls))
    idx_por_seed = {int(s): i for i, s in enumerate(seeds)}
    epsilon_por_seed: list = []
    n_pasos_cal: list = []
    lam_por_alpha_seed: dict = {float(a): [] for a in alphas}
    tau_por_alpha_seed: dict = {float(a): [] for a in alphas}
    riesgo_pares_por_alpha: dict = {float(a): [] for a in alphas}
    riesgo_pares_por_alpha_seed: dict = {float(a): {int(s): [] for s in seeds} for a in alphas}
    modelos: dict = {}
    epsilons: dict = {}
    semillas_cal = [2000 + int(s) for s in seeds]
    semillas_test = [3000 + int(s) for s in seeds]
    semillas_mundo = [int(semilla_mundo(int(s))) for s in seeds]
    for s in seeds:
        eps_tr = generar_trazas(int(s), int(n_trazas), 0.02)
        tr, _ = split_episodios(eps_tr, int(s))
        modelo = Atisbo.entrenar(tr, seed=int(s), epocas=int(epocas))
        modelos[int(s)] = modelo
        eps_cal0 = generar_trazas(int(s), int(n_trazas), 0.0)
        _, te0 = split_episodios(eps_cal0, int(s))
        epsilon = float(calibrar_epsilon(modelo, te0))
        epsilons[int(s)] = epsilon
        epsilon_por_seed.append(epsilon)
        eps_cal = generar_trazas(2000 + int(s), int(n_cal), 0.02)
        s_cal, m_cal, _ = _scores_malas(modelo, eps_cal)
        n_pasos_cal.append(int(s_cal.size))
        eps_te = generar_trazas(3000 + int(s), int(n_test), 0.02)
        s_te, m_te, por_ep = _scores_malas(modelo, eps_te)
        for a in alphas:
            af = float(a)
            lam = float(lambda_hat(s_cal, m_cal, af))
            tau = float(tau_desde_lambda(lam))
            lam_por_alpha_seed[af].append(lam)
            tau_por_alpha_seed[af].append(tau)
            for ss, mm in por_ep:
                num_ep = sum(int(m) for sc, m in zip(ss, mm) if float(sc) <= float(lam))
                den_ep = int(len(ss))
                riesgo_pares_por_alpha[af].append((num_ep, den_ep))
                riesgo_pares_por_alpha_seed[af][int(s)].append((num_ep, den_ep))
    por_tasa: dict = {float(a): [] for a in alphas}
    por_ll: dict = {float(a): [] for a in alphas}
    por_ex: dict = {float(a): [] for a in alphas}
    por_pasos: dict = {float(a): [] for a in alphas}
    por_tasa_seed: dict = {float(a): {int(s): [] for s in seeds} for a in alphas}
    por_ll_seed: dict = {float(a): {int(s): [] for s in seeds} for a in alphas}
    por_ex_seed: dict = {float(a): {int(s): [] for s in seeds} for a in alphas}
    malas_tot: dict = {float(a): 0 for a in alphas}
    atisbo_tot: dict = {float(a): 0 for a in alphas}
    total_llamadas = 0
    total_peticiones = 0
    total_invalidas = 0
    truncada = False
    pasos_parcial = 0
    n_ok = 0
    inf_tiempo = 0.0
    inf_llamadas = 0
    for s in seeds:
        modelo = modelos[int(s)]
        epsilon = float(epsilons[int(s)])
        predictor = _Cronometro(modelo)
        for a in alphas:
            af = float(a)
            idx = idx_por_seed[int(s)]
            lam = lam_por_alpha_seed[af][idx]
            tau = tau_por_alpha_seed[af][idx]
            mundo = Mundo(seed=semilla_mundo(int(s)), p_perturb=float(p))
            if modo == "simulada":
                politica = PoliticaSimulada(seed=int(s))
            else:
                politica = politica_compartida
            for _ in range(int(n)):
                registro: list = []
                res = correr_episodio(predictor, politica, mundo, tau=float(tau), epsilon=float(epsilon), registro=registro)
                inf_tiempo += float(predictor.tiempo_s)
                inf_llamadas += int(predictor.llamadas)
                predictor.tiempo_s = 0.0
                predictor.llamadas = 0
                if bool(res["truncada"]):
                    truncada = True
                    pasos_parcial = int(res["pasos"])
                    total_llamadas += int(res["llamadas_llm"])
                    total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
                    total_invalidas += int(res["llm_invalidas"])
                    break
                por_tasa[af].append(float(res["tasa_malas_aceptadas"]))
                por_ll[af].append(float(res["llamadas_llm"]))
                por_ex[af].append(1.0 if res["exito"] else 0.0)
                por_pasos[af].append(float(res["pasos"]))
                por_tasa_seed[af][int(s)].append(float(res["tasa_malas_aceptadas"]))
                por_ll_seed[af][int(s)].append(float(res["llamadas_llm"]))
                por_ex_seed[af][int(s)].append(1.0 if res["exito"] else 0.0)
                malas_tot[af] += int(res["malas_aceptadas"])
                atisbo_tot[af] += int(int(res["pasos"]) - int(res["llamadas_llm"]))
                total_llamadas += int(res["llamadas_llm"])
                total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
                total_invalidas += int(res["llm_invalidas"])
                n_ok += 1
            if truncada:
                break
        if truncada:
            break
    por_alpha = []
    for a in alphas:
        af = float(a)
        lams = lam_por_alpha_seed[af]
        taus = tau_por_alpha_seed[af]
        lam_json = [None if float(v) == float("-inf") else float(v) for v in lams]
        tau_json = [None if float(v) == float("inf") else float(v) for v in taus]
        acepta = [bool(float(v) == float("-inf")) for v in lams]
        n_ep = int(len(por_ll[af]))
        rt = _cociente(riesgo_pares_por_alpha[af]) if riesgo_pares_por_alpha[af] else 0.0
        if n_ep:
            re_ = _media(por_tasa[af])
            ll = _media(por_ll[af])
            ex = _media(por_ex[af])
            pa = _media(por_pasos[af])
            if int(atisbo_tot[af]) > 0:
                tc = float(float(malas_tot[af]) / float(atisbo_tot[af]))
            else:
                tc = None
        else:
            re_ = None
            ll = None
            ex = None
            pa = None
            tc = None
        por_alpha.append({"alpha": float(af), "lambda": lam_json, "tau": tau_json, "acepta_nada": [bool(v) for v in acepta], "n_episodios": int(n_ep), "riesgo_trazas": float(rt), "riesgo_empirico": (None if re_ is None else float(re_)), "tasa_condicional": (None if tc is None else float(tc)), "llamadas_llm": (None if ll is None else float(ll)), "exito": (None if ex is None else float(ex)), "pasos": (None if pa is None else float(pa))})
    evaluadas = [s for s in seeds if any(len(por_ll_seed[float(a)][int(s)]) > 0 for a in alphas)]
    metricas = {}
    for a in alphas:
        af = float(a)
        key_rt = f"riesgo_trazas@{float(af)}"
        key_re = f"riesgo_empirico@{float(af)}"
        key_ll = f"llamadas_llm@{float(af)}"
        key_ex = f"exito@{float(af)}"
        if riesgo_pares_por_alpha[af]:
            b = _bootstrap_cociente(riesgo_pares_por_alpha[af], n_remuestreos=1000, seed=0)
            b["por_seed"] = [_cociente(riesgo_pares_por_alpha_seed[af][int(s)]) for s in seeds if riesgo_pares_por_alpha_seed[af][int(s)]]
            metricas[key_rt] = b
        if por_tasa[af]:
            b = bootstrap(por_tasa[af], n_remuestreos=1000, seed=0)
            b["por_seed"] = [_media(por_tasa_seed[af][int(s)]) for s in seeds if por_tasa_seed[af][int(s)]]
            metricas[key_re] = b
        if por_ll[af]:
            b = bootstrap(por_ll[af], n_remuestreos=1000, seed=0)
            b["por_seed"] = [_media(por_ll_seed[af][int(s)]) for s in seeds if por_ll_seed[af][int(s)]]
            metricas[key_ll] = b
        if por_ex[af]:
            b = bootstrap(por_ex[af], n_remuestreos=1000, seed=0)
            b["por_seed"] = [_media(por_ex_seed[af][int(s)]) for s in seeds if por_ex_seed[af][int(s)]]
            metricas[key_ex] = b
    if cliente is not None:
        tok_in = int(cliente.tokens_entrada)
        tok_out = int(cliente.tokens_salida)
        red_total = int(getattr(politica_compartida, "llamadas_red", 0))
    else:
        tok_in = 0
        tok_out = 0
        red_total = 0
    coste = (tok_in * PRECIO_IN_MUSD + tok_out * PRECIO_OUT_MUSD) / 1e6
    with open(e1_path, encoding="utf-8") as f:
        e1 = json.load(f)
    comp = {"llamadas_llm_media": float(e1["metricas"]["llamadas_llm"]["media"]), "exito_media": float(e1["exito_media"]), "latencia_ms_media": float(e1["metricas"]["latencia_ms"]["media"]), "modo": str(e1["llm"]["modo"]), "commit": str(e1["commit"])}
    ent005 = entrada_h1(por_alpha)
    if ent005 is not None and n_ok and ent005["llamadas_llm"] is not None and ent005["exito"] is not None:
        e3_ll = float(ent005["llamadas_llm"])
        e3_ex = float(ent005["exito"])
    else:
        e3_ll = 0.0
        e3_ex = 0.0
    e1_ll = float(comp["llamadas_llm_media"])
    e1_ex = float(comp["exito_media"])
    truncada_total = bool(truncada or e1.get("truncada", False))
    if ent005 is None:
        h1_raw, det = False, {"ratio_llamadas": 0.0, "umbral_ratio": 0.3, "delta_exito_pp": 0.0, "umbral_delta_pp": -5.0}
    else:
        h1_raw, det = decidir_h1(e3_ll, e3_ex, e1_ll, e1_ex, truncada_total)
    h1 = bool(n_ok > 0 and h1_raw)
    ratio = float(det["ratio_llamadas"])
    delta_pp = float(det["delta_exito_pp"])
    notas = ["La garantia E[L] <= alpha de CRC es exacta si los pasos son intercambiables; aqui calibracion y test solo son intercambiables por episodio (los pasos de un episodio dependen entre si), asi que es aproximada y mejora con el numero de episodios de calibracion (n_cal).", "El ic95 de riesgo_trazas fija lambda_hat, no incluye la variabilidad de la calibracion.", "La garantia de CRC es para el test intercambiable (riesgo_trazas); el lazo tiene otra distribucion de estados y ademas ejecuta las acciones j = 1 y 2 del bloque, que la calibracion no cubre, asi que riesgo_empirico puede pasar de alfa: se reporta, no se fuerza.", "El coste estimado cuenta tambien las llamadas servidas desde la cache.", "bootstrap 1000 remuestreos semilla 0 sobre medias por episodio; riesgo_trazas remuestrea episodios de test con reemplazo y recalcula el cociente."]
    if truncada:
        notas.append("corrida truncada por tope LLM: h1 false; el parcial queda fuera de las metricas y en pasos_parcial.")
    datos = {"experimento": "E3", "fecha_utc": fecha_utc(), "commit": commit_hash(), "seeds": [int(x) for x in seeds], "config": {"seeds": [int(x) for x in seeds], "seeds_evaluadas": [int(x) for x in evaluadas], "alphas": [float(x) for x in alphas], "n": int(n), "n_cal": int(n_cal), "n_test": int(n_test), "n_trazas": int(n_trazas), "epocas": int(epocas), "episodios_completados": int(n_ok), "p_perturb": float(p), "epsilon_por_seed": [float(x) for x in epsilon_por_seed], "n_pasos_cal_por_seed": [int(x) for x in n_pasos_cal], "semillas_calibracion": [int(x) for x in semillas_cal], "semillas_test": [int(x) for x in semillas_test], "semillas_mundo": [int(x) for x in semillas_mundo], "max_llm_calls": int(max_llm_calls), "modelo": MODELO, "rapido": bool(rapido)}, "por_alpha": por_alpha, "metricas": metricas, "llm": {"modo": modo, "modelo": MODELO, "llamadas": int(total_llamadas), "peticiones": int(total_peticiones), "tokens_entrada": int(tok_in), "tokens_salida": int(tok_out), "coste_estimado_usd": float(coste), "llamadas_red": int(red_total)}, "llamadas_llm": int(total_llamadas), "llm_invalidas": int(total_invalidas), "coste_estimado_usd": float(coste), "truncada": bool(truncada), "pasos_parcial": int(pasos_parcial), "comparacion_e1": comp, "h1": bool(h1), "h1_detalle": {"ratio_llamadas": float(ratio), "umbral_ratio": 0.3, "delta_exito_pp": float(delta_pp), "umbral_delta_pp": -5.0}, "notas": notas}
    escribir_json(out, datos)
    media_ms = (inf_tiempo / inf_llamadas * 1000.0) if inf_llamadas else 0.0
    print(f"[E3] modo={modo} llamadas={total_llamadas} peticiones={total_peticiones} llamadas_red={red_total} truncada={truncada}")
    print(f"[E3] inferencia_atisbo_s={inf_tiempo:.3f} llamadas_pred={inf_llamadas} media_ms={media_ms:.3f}")
    return str(out)


if __name__ == "__main__":
    main()
