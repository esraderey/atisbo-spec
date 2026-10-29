"""Render del mundo (§1): render() puro + a_mp4(). CLI de vídeo."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from atisbo.estado import H, W

CELL = 32
BOT_COLOR = (64, 64, 64)
COLORES = {"rojo": (255, 0, 0), "verde": (0, 180, 0), "azul": (0, 0, 255)}
FONDO = (255, 255, 255)
REJILLA = (210, 210, 210)
BLOQUEO_COLOR = (190, 150, 220)


def render(estado) -> np.ndarray:
    img = Image.new("RGB", (W * CELL, H * CELL), FONDO)
    d = ImageDraw.Draw(img)
    for y in range(H):
        for x in range(W):
            d.rectangle([x * CELL, y * CELL, x * CELL + CELL - 1, y * CELL + CELL - 1], outline=REJILLA)
    if estado.bloqueo is not None:
        lx, ly = estado.bloqueo
        d.rectangle([lx * CELL + 1, ly * CELL + 1, lx * CELL + CELL - 2, ly * CELL + CELL - 2],
                    fill=BLOQUEO_COLOR)
    for tipo, (cx, cy) in estado.cubos.items():
        d.rectangle([cx * CELL, cy * CELL, cx * CELL + CELL - 1, cy * CELL + CELL - 1],
                    outline=COLORES[tipo], width=4)
    for _id, tipo, x, y, _e in estado.objetos:
        d.rectangle([x * CELL + 8, y * CELL + 8, x * CELL + 24, y * CELL + 24], fill=COLORES[tipo])
    bx, by = estado.bot
    d.rectangle([bx * CELL + 2, by * CELL + 2, bx * CELL + CELL - 3, by * CELL + CELL - 3],
                fill=BOT_COLOR)
    if estado.carga is not None:
        d.rectangle([bx * CELL + 5, by * CELL + 5, bx * CELL + 13, by * CELL + 13],
                    fill=COLORES[estado.carga])
    return np.asarray(img, dtype=np.uint8)


def a_mp4(frames: list[np.ndarray], ruta: Path | str, fps: int = 10) -> None:
    import imageio.v2 as imageio

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(ruta), fps=fps, codec="libx264", macro_block_size=None) as w:
        for fr in frames:
            w.append_data(np.asarray(fr, dtype=np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="Renderiza un episodio a MP4. Aviso: carga pickle solo de trazas propias.")
    ap.add_argument("--episodio", type=Path, required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with open(a.episodio, "rb") as f:
        eps = pickle.load(f)
    ep = eps[a.idx]
    frames = [render(s) for s, _ in ep]
    a_mp4(frames, a.out)


if __name__ == "__main__":
    main()
