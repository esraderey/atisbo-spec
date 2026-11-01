"""Resumen derivado de results/ (O7). Solo stdlib: sin numeros a mano."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cargar(path: Path) -> dict:
    with open(str(path), encoding="utf-8") as f:
        return json.load(f)


def _por_alpha_por_valor(por_alpha, alpha: float) -> dict:
    for e in por_alpha:
        if abs(float(e["alpha"]) - float(alpha)) < 1e-12:
            return e
    raise ValueError(f"alpha {alpha} no encontrado")


def _celda_por_nombre(celdas, nombre: str) -> dict:
    for c in celdas:
        if str(c.get("nombre")) == nombre:
            return c
    raise ValueError(f"celda {nombre} no encontrada")


def _celda_por_p(celdas, p: float) -> dict:
    for c in celdas:
        if abs(float(c["config"]["p"]) - float(p)) < 1e-12:
            return c
    raise ValueError(f"celda p={p} no encontrada")


def _miles(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def resumen(rd: Path) -> dict:
    rd = Path(str(rd))
    e1 = _cargar(rd / "E1.json")
    e2 = _cargar(rd / "E2.json")
    e2s = _cargar(rd / "E2_sam.json")
    e3 = _cargar(rd / "E3.json")
    e4 = _cargar(rd / "E4.json")
    e5 = _cargar(rd / "E5.json")
    reg = _cargar(rd / "registro_E2_seed0.json")
    media_e1 = float(e1["metricas"]["llamadas_llm"]["media"])
    e1_ep = int(e1["config"]["episodios_completados"])
    e2_ep = int(e2["config"]["episodios_completados"])
    e2s_ep = int(e2s["config"]["episodios_completados"])

    def _sin(episodios: int) -> int:
        return int(round(float(episodios) * float(media_e1), -2))

    filas = []
    filas.append(
        {
            "experimento": "E1: Muse Spark decide cada paso",
            "episodios": int(e1_ep),
            "llamadas": int(e1["llamadas_llm"]),
            "por_episodio": float(e1["metricas"]["llamadas_llm"]["media"]),
            "exito": float(e1["metricas"]["exito"]["media"]),
            "coste_usd": float(e1["coste_estimado_usd"]),
            "sin_atisbo": None,
        }
    )
    filas.append(
        {
            "experimento": "E2: Atisbo, τ = 0.5",
            "episodios": int(e2_ep),
            "llamadas": int(e2["llamadas_llm"]),
            "por_episodio": float(e2["metricas"]["llamadas_llm"]["media"]),
            "exito": float(e2["metricas"]["exito"]["media"]),
            "coste_usd": float(e2["coste_estimado_usd"]),
            "sin_atisbo": int(_sin(e2_ep)),
        }
    )
    filas.append(
        {
            "experimento": "E2_sam: E2 con estado desde SAM 3.1",
            "episodios": int(e2s_ep),
            "llamadas": int(e2s["llamadas_llm"]),
            "por_episodio": float(e2s["metricas"]["llamadas_llm"]["media"]),
            "exito": float(e2s["metricas"]["exito"]["media"]),
            "coste_usd": float(e2s["coste_estimado_usd"]),
            "sin_atisbo": int(_sin(e2s_ep)),
        }
    )
    for alpha, etiqueta in [(0.01, "E3: CRC, α = 0.01"), (0.05, "E3: CRC, α = 0.05"), (0.1, "E3: CRC, α = 0.1")]:
        ent = _por_alpha_por_valor(e3["por_alpha"], float(alpha))
        n_ep = int(ent["n_episodios"])
        met_ll = e3["metricas"][f"llamadas_llm@{float(alpha)}"]
        met_ex = e3["metricas"][f"exito@{float(alpha)}"]
        llamadas = int(round(float(met_ll["media"]) * float(n_ep)))
        filas.append(
            {
                "experimento": str(etiqueta),
                "episodios": int(n_ep),
                "llamadas": int(llamadas),
                "por_episodio": float(met_ll["media"]),
                "exito": float(met_ex["media"]),
                "coste_usd": None,
                "sin_atisbo": int(_sin(n_ep)),
            }
        )
    e3_ep = int(sum(int(e["n_episodios"]) for e in e3["por_alpha"]))
    filas.append(
        {
            "experimento": "E3 en total",
            "episodios": int(e3_ep),
            "llamadas": int(e3["llamadas_llm"]),
            "por_episodio": None,
            "exito": None,
            "coste_usd": float(e3["coste_estimado_usd"]),
            "sin_atisbo": int(_sin(e3_ep)),
        }
    )
    e4_ep = int(sum(int(c["n_episodios"]) for c in e4["celdas"]))
    filas.append(
        {
            "experimento": "E4 en total",
            "episodios": int(e4_ep),
            "llamadas": int(e4["llm"]["llamadas"]),
            "por_episodio": None,
            "exito": None,
            "coste_usd": float(e4["llm"]["coste_estimado_usd"]),
            "sin_atisbo": int(_sin(e4_ep)),
        }
    )
    e5_ep = int(sum(int(c["n_episodios"]) for c in e5["celdas"]))
    filas.append(
        {
            "experimento": "E5 en total",
            "episodios": int(e5_ep),
            "llamadas": int(e5["llm"]["llamadas"]),
            "por_episodio": None,
            "exito": None,
            "coste_usd": float(e5["llm"]["coste_estimado_usd"]),
            "sin_atisbo": int(_sin(e5_ep)),
        }
    )
    llamadas_total = int(e1["llamadas_llm"]) + int(e2["llamadas_llm"]) + int(e2s["llamadas_llm"]) + int(e3["llamadas_llm"]) + int(e4["llm"]["llamadas"]) + int(e5["llm"]["llamadas"])
    coste_total = float(e1["coste_estimado_usd"]) + float(e2["coste_estimado_usd"]) + float(e2s["coste_estimado_usd"]) + float(e3["coste_estimado_usd"]) + float(e4["llm"]["coste_estimado_usd"]) + float(e5["llm"]["coste_estimado_usd"])
    coste = {
        "filas": filas,
        "llamadas_total": int(llamadas_total),
        "coste_total_usd": float(coste_total),
        "sin_atisbo_e2_e3": int(round(200.0 * float(media_e1), -2)),
        "con_atisbo_e2_e3": int(int(e2["llamadas_llm"]) + int(e3["llamadas_llm"])),
    }

    def _h1(det: dict) -> dict:
        ratio = float(det["ratio_llamadas"])
        return {
            "ratio_llamadas": float(ratio),
            "ahorro_pct": float(100.0 * (1.0 - float(ratio))),
            "delta_exito_pp": float(det["delta_exito_pp"]),
        }

    h1 = {
        "e2": _h1(e2["h1_detalle"]),
        "e2_sam": _h1(e2s["h1_detalle"]),
        "e3": _h1(e3["h1_detalle"]),
    }
    riesgo = {}
    for e in e3["por_alpha"]:
        a = float(e["alpha"])
        riesgo[str(a)] = {"alpha": a, "riesgo_empirico": float(e["riesgo_empirico"]), "veces": float(e["riesgo_empirico"]) / a}
    met_a001 = e3["metricas"][f"llamadas_llm@{float(0.01)}"]
    ahorro_alpha_001 = float(100.0 * (1.0 - float(met_a001["media"]) / float(media_e1)))
    sam = {
        "coste_llm_usd": float(e2s["llm"]["coste_estimado_usd"]),
        "coste_sam_usd": float(e2s["sam"]["coste_estimado_usd"]),
        "imagenes": int(e2s["sam"]["imagenes"]),
        "coincidencia_bot_pct": float(float(e2s["metricas"]["coincidencia_bot"]["media"]) * 100.0),
        "coincidencia_todo_pct": float(float(e2s["metricas"]["coincidencia_todo"]["media"]) * 100.0),
    }
    pasos = list(reg["pasos"])
    n_pasos = int(len(pasos))
    n_llm = int(reg["llamadas_llm"])
    video = {
        "episodio": int(reg["episodio"]),
        "pasos": int(n_pasos),
        "llamadas_llm": int(n_llm),
        "pct_llm": float(100.0 * float(n_llm) / float(n_pasos)),
    }
    return {
        "coste": coste,
        "h1": h1,
        "riesgo": riesgo,
        "ahorro_alpha_001_pct": float(ahorro_alpha_001),
        "sam": sam,
        "video": video,
    }


def _fila_resultados(etiqueta: str, episodios: int, met_ll: dict, met_ex: dict) -> str:
    ll = float(met_ll["media"])
    lo, hi = met_ll["ic95"]
    ex = float(met_ex["media"])
    elo, ehi = met_ex["ic95"]
    return f"| {etiqueta} | {int(episodios)} | {ll:.2f} | {float(lo):.2f}–{float(hi):.2f} | {ex:.2f} | {float(elo):.2f}–{float(ehi):.2f} |"


def _tabla_resultados(e1: dict, e2: dict, e2s: dict, e3: dict, e4: dict, e5: dict) -> str:
    lineas = [
        "| Experimento | Episodios | Llamadas por episodio | IC95 | Éxito | IC95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lineas.append(_fila_resultados("E1: Muse Spark decide cada paso", int(e1["config"]["episodios_completados"]), e1["metricas"]["llamadas_llm"], e1["metricas"]["exito"]))
    lineas.append(_fila_resultados("E2: Atisbo, τ = 0.5", int(e2["config"]["episodios_completados"]), e2["metricas"]["llamadas_llm"], e2["metricas"]["exito"]))
    lineas.append(_fila_resultados("E2_sam: E2 con estado desde SAM 3.1", int(e2s["config"]["episodios_completados"]), e2s["metricas"]["llamadas_llm"], e2s["metricas"]["exito"]))
    for alpha, etiqueta in [(0.01, "E3: CRC, α = 0.01"), (0.05, "E3: CRC, α = 0.05"), (0.1, "E3: CRC, α = 0.1")]:
        ent = _por_alpha_por_valor(e3["por_alpha"], float(alpha))
        lineas.append(_fila_resultados(str(etiqueta), int(ent["n_episodios"]), e3["metricas"][f"llamadas_llm@{float(alpha)}"], e3["metricas"][f"exito@{float(alpha)}"]))
    c_sin = _celda_por_nombre(e4["celdas"], "sin_sorpresa")
    c_h1 = _celda_por_nombre(e4["celdas"], "horizonte_1")
    lineas.append(_fila_resultados("E4: sin sorpresa", int(c_sin["n_episodios"]), c_sin["metricas"]["llamadas_llm"], c_sin["metricas"]["exito"]))
    lineas.append(_fila_resultados("E4: horizonte 1", int(c_h1["n_episodios"]), c_h1["metricas"]["llamadas_llm"], c_h1["metricas"]["exito"]))
    c_p0 = _celda_por_p(e5["celdas"], 0.0)
    c_p005 = _celda_por_p(e5["celdas"], 0.05)
    lineas.append(_fila_resultados("E5: p = 0", int(c_p0["n_episodios"]), c_p0["metricas"]["llamadas_llm"], c_p0["metricas"]["exito"]))
    lineas.append(_fila_resultados("E5: p = 0.05", int(c_p005["n_episodios"]), c_p005["metricas"]["llamadas_llm"], c_p005["metricas"]["exito"]))
    return "\n".join(lineas)


def _tabla_riesgo(e3: dict) -> str:
    lineas = [
        "| α | τ por semilla (mín–máx) | Riesgo en trazas | IC95 | Riesgo en lazo cerrado | IC95 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for e in sorted(list(e3["por_alpha"]), key=lambda x: float(x["alpha"])):
        a = float(e["alpha"])
        taus = [float(v) for v in list(e["tau"]) if v is not None]
        tmin = float(min(taus))
        tmax = float(max(taus))
        rt = e3["metricas"][f"riesgo_trazas@{float(a)}"]
        re_ = e3["metricas"][f"riesgo_empirico@{float(a)}"]
        rt_lo, rt_hi = rt["ic95"]
        re_lo, re_hi = re_["ic95"]
        lineas.append(f"| {float(a):.2f} | {tmin:.2f}–{tmax:.2f} | {float(rt['media']):.4f} | {float(rt_lo):.4f}–{float(rt_hi):.4f} | {float(re_['media']):.3f} | {float(re_lo):.3f}–{float(re_hi):.3f} |")
    return "\n".join(lineas)


def _tabla_coste(derivado: dict) -> str:
    lineas = [
        "| Experimento | Episodios | Llamadas al LLM | Por episodio | Éxito | Coste estimado | Sin Atisbo (estimado) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f in derivado["coste"]["filas"]:
        por_txt = "" if f["por_episodio"] is None else f"{float(f['por_episodio']):.2f}"
        ex_txt = "" if f["exito"] is None else f"{float(f['exito']):.2f}"
        coste_txt = "—" if f["coste_usd"] is None else f"{float(f['coste_usd']):.2f} $"
        sin_txt = "—" if f["sin_atisbo"] is None else f"≈ {_miles(int(f['sin_atisbo']))}"
        lineas.append(f"| {f['experimento']} | {int(f['episodios'])} | {int(f['llamadas'])} | {por_txt} | {ex_txt} | {coste_txt} | {sin_txt} |")
    return "\n".join(lineas)


def tablas(rd: Path) -> str:
    rd = Path(str(rd))
    e1 = _cargar(rd / "E1.json")
    e2 = _cargar(rd / "E2.json")
    e2s = _cargar(rd / "E2_sam.json")
    e3 = _cargar(rd / "E3.json")
    e4 = _cargar(rd / "E4.json")
    e5 = _cargar(rd / "E5.json")
    derivado = resumen(rd)
    t1 = _tabla_resultados(e1, e2, e2s, e3, e4, e5)
    t2 = _tabla_riesgo(e3)
    t3 = _tabla_coste(derivado)
    return f"{t1}\n\n{t2}\n\n{t3}"


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Resumen derivado de results.")
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--out", type=str, default="results/resumen.json")
    return ap.parse_args(argv)


def main(argv=None) -> str:
    sys.stdout.reconfigure(encoding="utf-8")
    from atisbo.exp.comun import escribir_json

    a = _parse_args(argv)
    rd = Path(str(a.results))
    derivado = resumen(rd)
    escribir_json(str(a.out), derivado)
    print(tablas(rd))
    return str(a.out)


if __name__ == "__main__":
    main()
