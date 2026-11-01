"""Experimento E2 (PLIEGO 9): lazo especulativo con Atisbo + LLM."""
from __future__ import annotations

import argparse
import json
import os
import time

from atisbo.exp.comun import bootstrap, commit_hash, escribir_json, fecha_utc, generar_trazas, split_episodios
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import MODELO, ClienteMeta
from atisbo.modelo import Atisbo
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada
from atisbo.sam import MODELO_SAM, PROMPT, ClienteSam, MundoSam, PredictorCiego, TopeSam, mascaras_por_frame
from atisbo.sorpresa import calibrar_epsilon

PRECIO_IN_MUSD = 1.25
PRECIO_OUT_MUSD = 4.25
TOL_H1 = 1e-9


def decidir_h1(llamadas_e2, exito_e2, llamadas_e1, exito_e1, truncada):
    """Regla h1 pura: ratio <= 0.3 y delta >= -5pp, con tolerancia; false si truncado o E1 sin llamadas."""
    e2_ll = float(llamadas_e2)
    e2_ex = float(exito_e2)
    e1_ll = float(llamadas_e1)
    e1_ex = float(exito_e1)
    ratio = float(e2_ll / e1_ll) if e1_ll > 0 else 0.0
    delta_pp = float((e2_ex - e1_ex) * 100.0)
    detalle = {"ratio_llamadas": float(ratio), "umbral_ratio": 0.3, "delta_exito_pp": float(delta_pp), "umbral_delta_pp": -5.0}
    if bool(truncada):
        return False, detalle
    if not e1_ll > 0:
        return False, detalle
    ok = bool(ratio <= 0.3 + TOL_H1 and delta_pp >= -5.0 - TOL_H1)
    return ok, detalle


def semilla_mundo(s: int) -> int:
    """Semilla del Mundo para evaluar: 1000 + s, fuera del entrenamiento."""
    return 1000 + int(s)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="E2: lazo con Atisbo (tau fijo).")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-trazas", type=int, default=200)
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--max-llm-calls", type=int, default=1500)
    ap.add_argument("--cache", type=str, default="cache/llm")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--e1", type=str, default="results/E1.json")
    ap.add_argument("--modo", type=str, default=None)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--estado", type=str, default="sim")
    ap.add_argument("--cache-sam", type=str, default="cache/sam")
    ap.add_argument("--max-sam-calls", type=int, default=4000)
    ap.add_argument("--e2", type=str, default="results/E2.json")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    n = int(a.n)
    n_trazas = int(a.n_trazas)
    epocas = int(a.epocas)
    rapido = bool(a.rapido)
    if rapido:
        seeds = [seeds[0] if seeds else 0]
        n_trazas = 20
        epocas = 2
        n = 2
    estado = str(a.estado)
    if estado not in ("sim", "sam"):
        raise ValueError("estado sim|sam")
    out = str(a.out) if a.out is not None else ("results/E2_sam.json" if estado == "sam" else "results/E2.json")
    return seeds, float(a.p), float(a.tau), n, n_trazas, epocas, int(a.max_llm_calls), str(a.cache), out, str(a.e1), a.modo, rapido, estado, str(a.cache_sam), int(a.max_sam_calls), str(a.e2)


def _media(vals) -> float:
    vals = list(vals)
    return float(sum(vals) / len(vals)) if vals else 0.0


class _Cronometro:
    """Envuelve predecir para medir pared sin tocar el JSON."""

    def __init__(self, modelo) -> None:
        self._m = modelo
        self.tiempo_s = 0.0
        self.llamadas = 0

    def predecir(self, ventana):
        t0 = time.perf_counter()
        r = self._m.predecir(ventana)
        self.tiempo_s += time.perf_counter() - t0
        self.llamadas += 1
        return r


def main(argv=None, transporte=None, mascaras_de=None) -> str:
    seeds, p, tau, n, n_trazas, epocas, max_llm_calls, cache, out, e1_path, modo_arg, rapido, estado, cache_sam, max_sam_calls, e2_path = _parse_args(argv)
    modo = modo_arg if modo_arg is not None else os.environ.get("ATISBO_LLM", "simulada")
    if modo not in ("api", "replay", "simulada"):
        raise ValueError("modo api|replay|simulada")
    sam_on = estado == "sam"
    ref_e2 = None
    if sam_on:
        with open(e2_path, encoding="utf-8") as f0:
            ref_e2 = json.load(f0)
        eval0 = set(ref_e2["config"]["seeds_evaluadas"])
        for ss in seeds:
            if int(ss) not in eval0:
                raise ValueError("semilla %d no en seeds_evaluadas de %s" % (int(ss), e2_path))
    cliente_sam = None
    if sam_on:
        if modo in ("api", "replay"):
            cliente_sam = ClienteSam(modo=modo, cache_dir=cache_sam, transporte=transporte, max_red=max_sam_calls)
            if mascaras_de is None:
                mascaras_de = lambda frame: mascaras_por_frame(cliente_sam.segmentar([frame]), "imagen")[0]
        else:
            if mascaras_de is None:
                raise ValueError("estado sam con modo simulada exige mascaras_de")
    if modo == "simulada":
        cliente = None
        politica_compartida = None
    else:
        cliente = ClienteMeta(modo=modo, cache_dir=cache, transporte=transporte)
        politica_compartida = PoliticaLLM(cliente=cliente, max_llm_calls=int(max_llm_calls))
    por_llamadas: list = []
    por_exito: list = []
    por_racha: list = []
    por_tasa: list = []
    por_lat: list = []
    por_pasos: list = []
    por_malas: list = []
    por_xsorpresa: list = []
    por_coin_bot: list = []
    por_coin_obj: list = []
    por_coin_todo: list = []
    por_ciegos: list = []
    por_malas_real: list = []
    por_tasa_real: list = []
    por_semilla: dict = {s: {"llamadas": [], "exito": [], "racha": [], "tasa": [], "lat": [], "pasos": [], "malas": [], "xsorpresa": [], "coin_bot": [], "coin_obj": [], "coin_todo": [], "ciegos": [], "malas_real": [], "tasa_real": []} for s in seeds}
    epsilon_por_seed: list = []
    total_llamadas = 0
    total_peticiones = 0
    total_invalidas = 0
    truncada = False
    truncada_sam = False
    falsos_exitos = 0
    exitos_no_detectados = 0
    pasos_parcial = 0
    n_ok = 0
    inf_tiempo = 0.0
    inf_llamadas = 0
    for s in seeds:
        eps_tr = generar_trazas(s, n_trazas, 0.02)
        tr, _ = split_episodios(eps_tr, s)
        modelo = Atisbo.entrenar(tr, seed=s, epocas=epocas)
        eps_cal = generar_trazas(s, n_trazas, 0.0)
        _, te0 = split_episodios(eps_cal, s)
        epsilon = float(calibrar_epsilon(modelo, te0))
        epsilon_por_seed.append(epsilon)
        cronometro = _Cronometro(modelo)
        if sam_on:
            mundo = MundoSam(Mundo(seed=semilla_mundo(s), p_perturb=float(p)), mascaras_de)
            predictor = PredictorCiego(cronometro, mundo)
        else:
            predictor = cronometro
            mundo = Mundo(seed=semilla_mundo(s), p_perturb=float(p))
        if modo == "simulada":
            politica = PoliticaSimulada(seed=int(s))
        else:
            politica = politica_compartida
        for _ in range(int(n)):
            registro: list = []
            try:
                res = correr_episodio(predictor, politica, mundo, tau=float(tau), epsilon=float(epsilon), registro=registro)
            except TopeSam:
                truncada = True
                truncada_sam = True
                pasos_parcial = int(mundo.frames)
                break
            inf_tiempo += float(cronometro.tiempo_s)
            inf_llamadas += int(cronometro.llamadas)
            cronometro.tiempo_s = 0.0
            cronometro.llamadas = 0
            if bool(res["truncada"]):
                truncada = True
                pasos_parcial = int(res["pasos"])
                total_llamadas += int(res["llamadas_llm"])
                total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
                total_invalidas += int(res["llm_invalidas"])
                break
            exito_ep = bool(mundo.exito) if sam_on else bool(res["exito"])
            por_llamadas.append(float(res["llamadas_llm"]))
            por_exito.append(1.0 if exito_ep else 0.0)
            por_racha.append(float(res["racha_media"]))
            por_tasa.append(float(res["tasa_malas_aceptadas"]))
            por_pasos.append(float(res["pasos"]))
            por_malas.append(float(res["malas_aceptadas"]))
            por_xsorpresa.append(float(res["llamadas_por_sorpresa"]))
            lat_media = float(sum(float(e.get("latencia_ms", 0.0)) for e in registro) / len(registro)) if registro else 0.0
            por_lat.append(float(lat_media))
            por_semilla[s]["llamadas"].append(float(res["llamadas_llm"]))
            por_semilla[s]["exito"].append(1.0 if exito_ep else 0.0)
            por_semilla[s]["racha"].append(float(res["racha_media"]))
            por_semilla[s]["tasa"].append(float(res["tasa_malas_aceptadas"]))
            por_semilla[s]["pasos"].append(float(res["pasos"]))
            por_semilla[s]["malas"].append(float(res["malas_aceptadas"]))
            por_semilla[s]["xsorpresa"].append(float(res["llamadas_por_sorpresa"]))
            por_semilla[s]["lat"].append(float(lat_media))
            if sam_on:
                malas_real = sum(1 for ee, rr in zip(registro, mundo.rutinas_reales, strict=True) if ee.get("fuente") == "atisbo" and ee.get("accion") != rr)
                tasa_real = float(malas_real / len(registro)) if registro else 0.0
                por_malas_real.append(float(malas_real))
                por_tasa_real.append(float(tasa_real))
                por_semilla[s]["malas_real"].append(float(malas_real))
                por_semilla[s]["tasa_real"].append(float(tasa_real))
                if bool(res["exito"]) and not bool(mundo.exito):
                    falsos_exitos += 1
                if not bool(res["exito"]) and bool(mundo.exito):
                    exitos_no_detectados += 1
                coinc = mundo.coincidencias()
                por_coin_bot.append(float(coinc["bot"]))
                por_coin_obj.append(float(coinc["objetos"]))
                por_coin_todo.append(float(coinc["todo"]))
                por_ciegos.append(float(mundo.pasos_ciegos))
                por_semilla[s]["coin_bot"].append(float(coinc["bot"]))
                por_semilla[s]["coin_obj"].append(float(coinc["objetos"]))
                por_semilla[s]["coin_todo"].append(float(coinc["todo"]))
                por_semilla[s]["ciegos"].append(float(mundo.pasos_ciegos))
            total_llamadas += int(res["llamadas_llm"])
            total_peticiones += int(res.get("peticiones", res["llamadas_llm"]))
            total_invalidas += int(res["llm_invalidas"])
            n_ok += 1
        if truncada:
            break
    evaluadas = [s for s in seeds if por_semilla[s]["llamadas"]]
    if n_ok:
        metricas = {}
        for clave, vals, sk in (
            ("llamadas_llm", por_llamadas, "llamadas"),
            ("exito", por_exito, "exito"),
            ("racha_media", por_racha, "racha"),
            ("tasa_malas_aceptadas", por_tasa, "tasa"),
            ("pasos", por_pasos, "pasos"),
            ("malas_aceptadas", por_malas, "malas"),
            ("llamadas_por_sorpresa", por_xsorpresa, "xsorpresa"),
            ("latencia_ms", por_lat, "lat"),
        ):
            b = bootstrap(vals, n_remuestreos=1000, seed=0)
            b["por_seed"] = [_media(por_semilla[s][sk]) for s in evaluadas]
            metricas[clave] = b
        if sam_on:
            for clave, vals, sk in (
                ("coincidencia_bot", por_coin_bot, "coin_bot"),
                ("coincidencia_objetos", por_coin_obj, "coin_obj"),
                ("coincidencia_todo", por_coin_todo, "coin_todo"),
                ("pasos_ciegos", por_ciegos, "ciegos"),
                ("malas_aceptadas_real", por_malas_real, "malas_real"),
                ("tasa_malas_aceptadas_real", por_tasa_real, "tasa_real"),
            ):
                b = bootstrap(vals, n_remuestreos=1000, seed=0)
                b["por_seed"] = [_media(por_semilla[s][sk]) for s in evaluadas]
                metricas[clave] = b
    else:
        metricas = {}
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
    comp = {
        "llamadas_llm_media": float(e1["metricas"]["llamadas_llm"]["media"]),
        "exito_media": float(e1["exito_media"]),
        "latencia_ms_media": float(e1["metricas"]["latencia_ms"]["media"]),
        "modo": str(e1["llm"]["modo"]),
        "commit": str(e1["commit"]),
    }
    e2_ll = float(metricas["llamadas_llm"]["media"]) if n_ok else 0.0
    e2_ex = float(sum(por_exito) / len(por_exito)) if por_exito else 0.0
    e1_ll = float(comp["llamadas_llm_media"])
    e1_ex = float(comp["exito_media"])
    truncada_total = bool(truncada or e1.get("truncada", False))
    h1_raw, det = decidir_h1(e2_ll, e2_ex, e1_ll, e1_ex, truncada_total)
    h1 = bool(n_ok > 0 and h1_raw)
    ratio = float(det["ratio_llamadas"])
    delta_pp = float(det["delta_exito_pp"])
    notas = [
        "E2 lazo especulativo con Atisbo entrenado en p 0.02; epsilon calibrado por semilla en test p 0.0; lazo en Mundo(1000+s).",
        "comparacion_e1: racha_media y tasa_malas_aceptadas de E1 valen 0 por construccion (todo LLM).",
        "h1 aqui es con tau fijo 0.5; con tau conforme (alfa = 0.05) es E3.",
        "bootstrap 1000 remuestreos semilla 0 sobre medias por episodio.",
    ]
    if truncada:
        notas.append("corrida truncada por tope LLM: h1 false; el parcial queda fuera de las metricas y en pasos_parcial.")
    if sam_on:
        notas.append("malas_aceptadas mide contra la rutina percibida y malas_aceptadas_real contra la real.")
    if truncada_sam:
        notas.append("corrida truncada por tope SAM")
    config = {
        "seeds": [int(x) for x in seeds],
        "seeds_evaluadas": [int(x) for x in evaluadas],
        "n_episodios_por_seed": int(n),
        "episodios_completados": int(n_ok),
        "p_perturb": float(p),
        "tau": float(tau),
        "epsilon_por_seed": [float(x) for x in epsilon_por_seed],
        "epocas": int(epocas),
        "n_trazas": int(n_trazas),
        "semillas_mundo": [int(semilla_mundo(s)) for s in seeds],
        "max_llm_calls": int(max_llm_calls),
        "modelo": MODELO,
        "rapido": bool(rapido),
    }
    datos = {
        "experimento": "E2",
        "fecha_utc": fecha_utc(),
        "commit": commit_hash(),
        "seeds": [int(x) for x in seeds],
        "config": config,
        "metricas": metricas,
        "llm": {"modo": modo, "modelo": MODELO, "llamadas": int(total_llamadas), "peticiones": int(total_peticiones), "tokens_entrada": int(tok_in), "tokens_salida": int(tok_out), "coste_estimado_usd": float(coste)},
        "llamadas_llm": int(total_llamadas),
        "exito_media": float(e2_ex),
        "llm_invalidas": int(total_invalidas),
        "coste_estimado_usd": float(coste),
        "truncada": bool(truncada),
        "pasos_parcial": int(pasos_parcial),
        "comparacion_e1": comp,
        "h1": bool(h1),
        "h1_detalle": {"ratio_llamadas": float(ratio), "umbral_ratio": 0.3, "delta_exito_pp": float(delta_pp), "umbral_delta_pp": -5.0},
        "notas": notas,
    }
    if sam_on:
        config["estado"] = "sam"
        config["prompt_sam"] = PROMPT
        config["max_sam_calls"] = int(max_sam_calls)
        if cliente_sam is not None:
            sam_peticiones = int(cliente_sam.peticiones)
            sam_imagenes = int(cliente_sam.imagenes)
            sam_red = int(cliente_sam.llamadas_red)
            sam_coste = float(cliente_sam.coste_estimado_usd())
            sam_lat = float(cliente_sam.latencia_ms_media())
        else:
            sam_peticiones = 0
            sam_imagenes = 0
            sam_red = 0
            sam_coste = 0.0
            sam_lat = 0.0
        datos["sam"] = {"modo": modo, "modelo": MODELO_SAM, "prompt": PROMPT, "peticiones": sam_peticiones, "imagenes": sam_imagenes, "llamadas_red": sam_red, "coste_estimado_usd": float(sam_coste), "falsos_exitos": int(falsos_exitos), "exitos_no_detectados": int(exitos_no_detectados), "latencia_ms_media": float(sam_lat)}
        datos["coste_estimado_usd"] = float(coste + sam_coste)
        if evaluadas:
            ref_eval = list(ref_e2["config"]["seeds_evaluadas"])
            ref_ll = list(ref_e2["metricas"]["llamadas_llm"]["por_seed"])
            ref_ex = list(ref_e2["metricas"]["exito"]["por_seed"])
            ll_sel = []
            ex_sel = []
            for s in evaluadas:
                idx = ref_eval.index(s)
                ll_sel.append(float(ref_ll[idx]))
                ex_sel.append(float(ref_ex[idx]))
            cfg = ref_e2["config"]
            comparable = bool(cfg["p_perturb"] == float(p) and cfg["tau"] == float(tau) and cfg["n_episodios_por_seed"] == int(n) and cfg["n_trazas"] == int(n_trazas) and cfg["epocas"] == int(epocas) and cfg["rapido"] is False and rapido is False)
            datos["comparacion_e2"] = {
                "llamadas_llm_media": float(sum(ll_sel) / len(ll_sel)),
                "exito_media": float(sum(ex_sel) / len(ex_sel)),
                "commit": str(ref_e2["commit"]),
                "modo": str(ref_e2["llm"]["modo"]),
                "comparable": bool(comparable),
            }
        datos["notas"] = list(notas) + [
            "estado sam: el lazo solo ve lo que reconstruye SAM desde cada frame.",
            "estado sam: exito se mide en el mundo real, no en lo percibido.",
            "estado sam: un frame sin bot fuerza LLM al empezar un bloque, y dentro de un bloque lo decide la sorpresa.",
            "estado sam: el coste suma LLM y SAM y cuenta lo servido de caché.",
        ]
    escribir_json(out, datos)
    media_ms = (inf_tiempo / inf_llamadas * 1000.0) if inf_llamadas else 0.0
    print(f"[E2] modo={modo} llamadas={total_llamadas} peticiones={total_peticiones} llamadas_red={red_total} truncada={truncada}")
    print(f"[E2] inferencia_atisbo_s={inf_tiempo:.3f} llamadas_pred={inf_llamadas} media_ms={media_ms:.3f}")
    if sam_on:
        todo_m = float(metricas["coincidencia_todo"]["media"]) if n_ok else 0.0
        sam_b = datos["sam"]
        print(f"[E2-sam] imagenes={sam_b['imagenes']} llamadas_red={sam_b['llamadas_red']} coste={datos['coste_estimado_usd']:.6f} coincidencia_todo={todo_m:.3f}")
    return str(out)


if __name__ == "__main__":
    main()
