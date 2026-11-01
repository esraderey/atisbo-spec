"""Cuatro figuras desde results/*.json (O7 parte 1). Matplotlib en backend Agg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _cargar(path: Path) -> dict:
    with open(str(path), encoding="utf-8") as f:
        return json.load(f)


def _media_ic(met: dict) -> tuple:
    media = float(met["media"])
    lo, hi = met["ic95"]
    return media, float(lo), float(hi)


def _err(medias, los, his) -> list:
    lower = [max(float(m) - float(lo), 0.0) for m, lo in zip(medias, los)]
    upper = [max(float(hi) - float(m), 0.0) for m, hi in zip(medias, his)]
    return [lower, upper]


def _barras_ic(ax, xs, medias, los, his, **kw):
    return ax.bar(list(xs), list(medias), yerr=_err(medias, los, his), capsize=4, **kw)


def _fig_llamadas_vs_alpha(e1, e2, e3, out: Path) -> None:
    alphas = sorted(float(e["alpha"]) for e in e3["por_alpha"])
    medias, los, his = [], [], []
    for a in alphas:
        met = e3["metricas"][f"llamadas_llm@{float(a)}"]
        m, lo, hi = _media_ic(met)
        medias.append(m)
        los.append(lo)
        his.append(hi)
    e1m, _, _ = _media_ic(e1["metricas"]["llamadas_llm"])
    e2m, _, _ = _media_ic(e2["metricas"]["llamadas_llm"])
    fig, ax = plt.subplots()
    _barras_ic(ax, [str(a) for a in alphas], medias, los, his)
    ax.axhline(float(e1m), linestyle="--", label="E1")
    ax.axhline(float(e2m), linestyle=":", label="E2")
    ax.set_xlabel("α")
    ax.set_ylabel("llamadas por episodio")
    ax.set_title("Llamadas por α (E3) frente a E1 y E2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out))
    plt.close(fig)


def _fig_riesgo_vs_alpha(e3, out: Path) -> None:
    alphas = sorted(float(e["alpha"]) for e in e3["por_alpha"])
    rt_m, rt_lo, rt_hi, re_m, re_lo, re_hi = [], [], [], [], [], []
    for a in alphas:
        m, lo, hi = _media_ic(e3["metricas"][f"riesgo_trazas@{float(a)}"])
        rt_m.append(m)
        rt_lo.append(lo)
        rt_hi.append(hi)
        m2, lo2, hi2 = _media_ic(e3["metricas"][f"riesgo_empirico@{float(a)}"])
        re_m.append(m2)
        re_lo.append(lo2)
        re_hi.append(hi2)
    x = np.asarray(alphas, dtype=float)
    fig, ax = plt.subplots()
    ax.errorbar(x, rt_m, yerr=_err(rt_m, rt_lo, rt_hi), fmt="o-", capsize=4, label="riesgo_trazas")
    ax.errorbar(x, re_m, yerr=_err(re_m, re_lo, re_hi), fmt="s-", capsize=4, label="riesgo_empirico")
    ax.plot(x, x, "k--", label="y = α")
    ax.set_xlabel("α")
    ax.set_ylabel("riesgo")
    ax.set_title("Riesgo E3 por α")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out))
    plt.close(fig)


def _fig_e1_vs_e2(e1, e2, e2sam, out: Path) -> None:
    nombres = ["E1", "E2"]
    datos = [e1, e2]
    if e2sam is not None:
        semillas = ", ".join(str(s) for s in e2sam["seeds"])
        nombres.append(f"E2_sam (semillas {semillas})")
        datos.append(e2sam)
    ll_m, ll_lo, ll_hi, ex_m, ex_lo, ex_hi = [], [], [], [], [], []
    for d in datos:
        m, lo, hi = _media_ic(d["metricas"]["llamadas_llm"])
        ll_m.append(m)
        ll_lo.append(lo)
        ll_hi.append(hi)
        m2, lo2, hi2 = _media_ic(d["metricas"]["exito"])
        ex_m.append(m2)
        ex_lo.append(lo2)
        ex_hi.append(hi2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    _barras_ic(a1, nombres, ll_m, ll_lo, ll_hi)
    a1.set_ylabel("llamadas por episodio")
    a1.set_title("Llamadas E1 vs E2")
    _barras_ic(a2, nombres, ex_m, ex_lo, ex_hi)
    a2.set_ylabel("éxito")
    a2.set_title("Éxito E1 vs E2")
    fig.tight_layout()
    fig.savefig(str(out))
    plt.close(fig)


def _fig_matriz_e5(e5, out: Path) -> None:
    celdas = [c for c in list(e5["celdas"]) if c.get("metricas")]
    celdas = sorted(celdas, key=lambda c: float(c["config"]["p"]))
    nombres = [f"p={float(c['config']['p'])}" + (" (parcial)" if c.get("parcial") else "") for c in celdas]
    ll_m, ll_lo, ll_hi, ex_m, ex_lo, ex_hi = [], [], [], [], [], []
    for c in celdas:
        m, lo, hi = _media_ic(c["metricas"]["llamadas_llm"])
        ll_m.append(m)
        ll_lo.append(lo)
        ll_hi.append(hi)
        m2, lo2, hi2 = _media_ic(c["metricas"]["exito"])
        ex_m.append(m2)
        ex_lo.append(lo2)
        ex_hi.append(hi2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    _barras_ic(a1, nombres, ll_m, ll_lo, ll_hi)
    a1.set_ylabel("llamadas")
    a1.set_title("E5 llamadas por p")
    _barras_ic(a2, nombres, ex_m, ex_lo, ex_hi)
    a2.set_ylabel("éxito")
    a2.set_title("E5 éxito por p")
    fig.tight_layout()
    fig.savefig(str(out))
    plt.close(fig)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Dibuja las cuatro figuras desde results.")
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--out", type=str, default="demo/out")
    return ap.parse_args(argv)


def main(argv=None) -> str:
    """Lee E1/E2/E3/E5 de --results y escribe los cuatro PNG en --out."""
    a = _parse_args(argv)
    rd = Path(str(a.results))
    outd = Path(str(a.out))
    outd.mkdir(parents=True, exist_ok=True)
    e1 = _cargar(rd / "E1.json")
    e2 = _cargar(rd / "E2.json")
    e3 = _cargar(rd / "E3.json")
    e5 = _cargar(rd / "E5.json")
    sam_path = rd / "E2_sam.json"
    e2sam = _cargar(sam_path) if sam_path.exists() else None
    _fig_llamadas_vs_alpha(e1, e2, e3, outd / "llamadas_vs_alpha.png")
    _fig_riesgo_vs_alpha(e3, outd / "riesgo_vs_alpha.png")
    _fig_e1_vs_e2(e1, e2, e2sam, outd / "e1_vs_e2.png")
    _fig_matriz_e5(e5, outd / "matriz_e5.png")
    return str(outd)


if __name__ == "__main__":
    main()
