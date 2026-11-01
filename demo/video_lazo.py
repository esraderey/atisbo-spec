"""Video anotado del lazo (O7 parte 1). Solo PIL y numpy, sin UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from atisbo.env.render import CELL, a_mp4, render
from atisbo.estado import H, W
from demo.registro import estado_desde_dict

ALTO_BANDA = 64
ANCHO = W * CELL
ALTO_MUNDO = H * CELL
MARCADOR = (16, ALTO_MUNDO + 10, 72, ALTO_MUNDO + 54)
VERDE = (0, 200, 0)
ROJO = (220, 0, 0)
_FANTASMA = (255, 255, 0, 110)
_DELTA = {"ARRIBA": (0, -1), "ABAJO": (0, 1), "IZQ": (-1, 0), "DER": (1, 0)}
_FUENTE = ImageFont.load_default(size=16)


def contadores(pasos) -> list:
    """Lista de (n_llm, n_pasos) acumulados por paso."""
    out: list = []
    n_llm = 0
    for i, p in enumerate(pasos):
        if p.get("fuente") == "llm":
            n_llm += 1
        out.append((int(n_llm), int(i + 1)))
    return out


def _celdas_fantasma(paso) -> list:
    estado = estado_desde_dict(paso["estado"])
    x, y = (int(estado.bot[0]), int(estado.bot[1]))
    celdas: list = []
    for nombre in list(paso.get("predichas", [])):
        dx, dy = _DELTA.get(str(nombre), (0, 0))
        nx = min(W - 1, max(0, x + dx))
        ny = min(H - 1, max(0, y + dy))
        if (nx, ny) != estado.bloqueo:
            x, y = nx, ny
        celdas.append((x, y))
    return celdas


def pintar_paso(paso, n_llm, n_pasos) -> np.ndarray:
    """Render del estado + fantasma + banda con marcador y contadores."""
    estado = estado_desde_dict(paso["estado"])
    base = render(estado)
    img = Image.fromarray(base).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay, "RGBA")
    for (cx, cy) in _celdas_fantasma(paso):
        d.rectangle(
            [cx * CELL + 2, cy * CELL + 2, cx * CELL + CELL - 3, cy * CELL + CELL - 3],
            fill=_FANTASMA,
        )
    img = Image.alpha_composite(img, overlay).convert("RGB")
    lienzo = Image.new("RGB", (ANCHO, ALTO_MUNDO + ALTO_BANDA), (20, 20, 20))
    lienzo.paste(img, (0, 0))
    dl = ImageDraw.Draw(lienzo)
    color = VERDE if str(paso.get("fuente")) == "atisbo" else ROJO
    dl.rectangle(list(MARCADOR), fill=color)
    dl.text((90, ALTO_MUNDO + 8), f"llamadas LLM: {int(n_llm)} / pasos: {int(n_pasos)}", fill=(255, 255, 255), font=_FUENTE)
    dl.text((90, ALTO_MUNDO + 30), str(paso.get("fuente")), fill=(255, 255, 255), font=_FUENTE)
    return np.asarray(lienzo, dtype=np.uint8)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Renderiza un registro del lazo a MP4.")
    ap.add_argument("--episodio", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fps", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> str:
    """Lee el JSON de registro, pinta cada paso y escribe el MP4."""
    a = _parse_args(argv)
    with open(str(a.episodio), encoding="utf-8") as f:
        datos = json.load(f)
    pasos = list(datos["pasos"])
    conteo = contadores(pasos)
    frames = [pintar_paso(p, n_llm, n_pasos) for p, (n_llm, n_pasos) in zip(pasos, conteo)]
    final = datos.get("estado_final")
    if final is not None and pasos:
        ultimo = pasos[-1]
        n_llm, n_pasos = conteo[-1]
        paso_final = dict(ultimo)
        paso_final["estado"] = final
        paso_final["predichas"] = []
        frames.append(pintar_paso(paso_final, n_llm, n_pasos))
    a_mp4(frames, Path(str(a.out)), fps=int(a.fps))
    return str(a.out)


if __name__ == "__main__":
    main()
