"""Consola en vivo: un episodio con llamadas reales a Muse Spark y SAM 3.1, visibles paso a paso.

Sirve solo en 127.0.0.1. La clave la leen los clientes de META_API_KEY y nunca llega a la página.
Atisbo no es un modelo de Meta y va apagado por defecto: sin --atisbo decide Muse Spark en cada paso.
Uso: python -m demo.vivo --modo api   (pide la clave sin eco si falta)
     python -m demo.vivo --modo simulada   (ensayo sin red ni modelos de Meta)
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import getpass
import io
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from atisbo.control import PredictorNulo
from atisbo.env.perturbaciones import celdas_vacias
from atisbo.env.render import BLOQUEO_COLOR, BOT_COLOR, CELL, COLORES, render
from atisbo.env.trazas import PRESUPUESTO
from atisbo.lazo import Mundo, correr_episodio
from atisbo.meta_api import MODELO, ClienteMeta
from atisbo.politica_llm import PoliticaLLM, PoliticaSimulada
from atisbo.sam import MODELO_SAM, PROMPT, ClienteSam, MundoSam, PredictorCiego, mascaras_por_frame

CUBO, BOT, OBJETO = (188, 140, 255), (232, 184, 72), (57, 211, 235)
TAU = 0.5
SUSTITUTO = "local stand-in"


def png(arr) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def capa_mascaras(frame, mascaras) -> np.ndarray:
    """Fotograma con cada máscara teñida según su tamaño: cubo, bot u objeto."""
    out = np.asarray(frame, dtype=np.float32).copy()
    for m in mascaras.values():
        m = np.asarray(m, dtype=bool)
        xs = np.nonzero(m.any(axis=0))[0]
        if xs.size == 0:
            continue
        w = xs[-1] - xs[0] + 1
        color = CUBO if w >= 30 else (BOT if w >= 24 else OBJETO)
        out[m] = out[m] * 0.14 + np.asarray(color, dtype=np.float32) * 0.86
    return out


def mascaras_por_color(frame) -> dict:
    """Máscaras exactas por color, sin modelo: solo para ensayar la consola sin gastar."""
    arr = np.asarray(frame, dtype=np.uint8)
    out: dict = {}
    for cy in range(12):
        for cx in range(12):
            y0, y1, x0, x1 = cy * CELL, (cy + 1) * CELL, cx * CELL, (cx + 1) * CELL
            for col in (BOT_COLOR, BLOQUEO_COLOR, *COLORES.values()):
                m = np.all(arr[y0:y1, x0:x1] == np.asarray(col, dtype=np.uint8), axis=2)
                if m.any():
                    lienzo = np.zeros((384, 384), dtype=bool)
                    lienzo[y0:y1, x0:x1] = m
                    out[len(out) + 1] = lienzo
    return out


class Bus:
    """Eventos numerados para la página. Nunca lleva cabeceras ni la clave."""

    def __init__(self) -> None:
        self._ev: list = []
        self._lock = threading.Lock()

    def publica(self, **ev) -> None:
        with self._lock:
            ev["n"] = len(self._ev)
            self._ev.append(ev)

    def desde(self, n: int) -> list:
        with self._lock:
            return list(self._ev[int(n):])


def mover_el_cercano(s):
    """Lleva el objeto libre más cercano al bot a la casilla vacía más lejana: así el cambio de plan se ve."""
    bx, by = s.bot
    dist = lambda x, y: abs(x - bx) + abs(y - by)
    libres = [o for o in s.objetos if not o[4]]
    vacias = celdas_vacias(s)
    if not libres or not vacias:
        return s, False
    obj = min(libres, key=lambda o: (dist(o[2], o[3]), o[0]))
    nx, ny = max(vacias, key=lambda c: dist(*c))
    objetos = tuple((o[0], o[1], nx, ny, o[4]) if o[0] == obj[0] else o for o in s.objetos)
    return dataclasses.replace(s, objetos=objetos), True


class MundoVivo:
    """El Mundo real, más una perturbación a petición de quien mira."""

    def __init__(self, interno, bus) -> None:
        self.interno = interno
        self.bus = bus
        self.pendiente = False

    @property
    def exito(self) -> bool:
        return self.interno.exito

    def perturbar(self) -> None:
        self.pendiente = True

    def reiniciar(self):
        self.pendiente = False
        return self.interno.reiniciar()

    def avanzar(self, s, a):
        antes = len(self.interno.perturbaciones)
        s2 = self.interno.avanzar(s, a)
        if len(self.interno.perturbaciones) > antes:
            self.bus.publica(tipo="perturbacion", ok=True, pedida=False)
        if not self.pendiente or self.interno.exito:
            return s2
        self.pendiente = False
        s3, ok = mover_el_cercano(s2)
        movidos = [[o[2], o[3]] for o in s3.objetos if o not in s2.objetos] if ok else []
        self.bus.publica(tipo="perturbacion", ok=bool(ok), pedida=True, celda=movidos[0] if movidos else None)
        return s3 if ok else s2


class _ClienteVisto:
    """Publica cada llamada real a Muse Spark tal como responde."""

    def __init__(self, cliente, bus) -> None:
        self.cliente = cliente
        self.bus = bus

    def __getattr__(self, nombre):
        return getattr(self.cliente, nombre)

    def completar(self, payload):
        texto, uso, lat = self.cliente.completar(payload)
        self.bus.publica(
            tipo="llamada", modelo=MODELO, ms=round(float(lat)), entrada=int(uso.get("prompt_tokens", uso.get("input_tokens", 0))), salida=int(uso.get("completion_tokens", uso.get("output_tokens", 0))),
            red=not self.cliente.ultima_desde_cache, respuesta=texto[:80],
        )
        return texto, uso, lat


class Consola:
    """Un episodio por vez, con cada decisión y cada llamada publicadas en el bus."""

    def __init__(self, predictor, politica, mundo, tau, epsilon, modo, bus=None, cliente_sam=None, mascaras_de=None, pausa=0.25, con_atisbo=True) -> None:
        self.bus = bus if bus is not None else Bus()
        self.vivo = MundoVivo(mundo, self.bus)
        self.tau, self.epsilon, self.modo, self.pausa = float(tau), float(epsilon), modo, float(pausa)
        self.politica = politica
        self.cliente_sam = cliente_sam
        self.mascaras_de = mascaras_de
        self.con_sam = cliente_sam is not None or mascaras_de is not None
        self.mundo = MundoSam(self.vivo, self._ve) if self.con_sam else self.vivo
        self.predictor = PredictorCiego(predictor, self.mundo) if self.con_sam else predictor
        self.corriendo = False
        self.registro: list = []
        self._frame = self._capa = None
        self._pred: list = []
        self.pasos = 0
        self.bus.publica(tipo="listo", modo=modo, sam=self.con_sam, atisbo=bool(con_atisbo), presupuesto=PRESUPUESTO)

    def _ve(self, frame):
        t0 = time.perf_counter()
        if self.cliente_sam is not None:
            m = mascaras_por_frame(self.cliente_sam.segmentar([frame]), "imagen")[0]
            detalle = {"modelo": MODELO_SAM, "prompt": PROMPT, "red": not self.cliente_sam.ultima_desde_cache}
        else:
            m = self.mascaras_de(frame)
            detalle = {"modelo": SUSTITUTO}
        self.bus.publica(tipo="llamada", ms=round((time.perf_counter() - t0) * 1000), mascaras=len(m), **detalle)
        self._frame, self._capa = np.asarray(frame), capa_mascaras(frame, m)
        return m

    def predecir(self, ventana):
        pred = self.predictor.predecir(ventana)
        self._pred = [a.name for a in pred.acciones]
        return pred

    def reiniciar(self):
        self.pasos = 0
        s = self.mundo.reiniciar()
        self._publica(s)
        return s

    def avanzar(self, s, a):
        e = self.registro[-1]
        s2 = self.mundo.avanzar(s, a)
        self.pasos += 1
        self._publica(s2, e)
        time.sleep(self.pausa)
        return s2

    def _publica(self, s, e=None) -> None:
        frame = self._frame if self.con_sam else render(s)
        ev = {"tipo": "paso", "paso": self.pasos, "frame": png(frame), "mascaras": png(self._capa) if self.con_sam else None}
        if e is not None:
            # NaN no es JSON válido: colgaría la página. El lazo ya lo trata como confianza baja.
            conf = float(e["conf"])
            ev.update(fuente=e["fuente"], accion=e["accion"], conf=conf if math.isfinite(conf) else -1.0, predichas=list(self._pred), sorpresa=e["fuente"] == "llm" and float(e["sorpresa"]) != 0.0)
        self.bus.publica(**ev)

    def episodio(self) -> None:
        self.bus.publica(tipo="inicio")
        self.registro = []
        try:
            res = correr_episodio(self, self.politica, self, tau=self.tau, epsilon=self.epsilon, registro=self.registro)
            self.bus.publica(tipo="fin", exito=bool(self.mundo.exito), pasos=int(res["pasos"]), llamadas=int(res["llamadas_llm"]), truncada=bool(res["truncada"]))
        except Exception as exc:
            # El hilo no tiene a quién avisar salvo la página; sin esto se quedaría esperando para siempre.
            clave = os.environ.get("META_API_KEY", "").strip()
            texto = str(exc)
            self.bus.publica(tipo="error", texto=texto.replace(clave, "***") if clave else texto)
        finally:
            self.corriendo = False

    def empezar(self) -> bool:
        if self.corriendo:
            return False
        self.corriendo = True
        threading.Thread(target=self.episodio, daemon=True).start()
        return True


def crear_servidor(consola, puerto: int) -> ThreadingHTTPServer:
    pagina = Path(__file__).with_name("vivo.html").read_bytes()

    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def _host_ok(self) -> bool:
            p = self.server.server_address[1]
            return self.headers.get("Host", "") in (f"127.0.0.1:{p}", f"localhost:{p}")

        def _responde(self, codigo, cuerpo, tipo="application/json") -> None:
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if not self._host_ok():
                self._responde(403, b"{}")
            elif url.path == "/":
                self._responde(200, pagina, "text/html; charset=utf-8")
            elif url.path == "/eventos":
                n = int(parse_qs(url.query).get("desde", ["0"])[0])
                self._responde(200, json.dumps(consola.bus.desde(n)).encode("utf-8"))
            else:
                self._responde(404, b"{}")

        def do_POST(self) -> None:
            if not self._host_ok() or self.headers.get("X-Consola") != "1":
                self._responde(403, b"{}")
            elif self.path == "/empezar":
                self._responde(200, json.dumps({"empezado": consola.empezar()}).encode("utf-8"))
            elif self.path == "/perturbar":
                consola.vivo.perturbar()
                self._responde(200, b"{}")
            else:
                self._responde(404, b"{}")

    return ThreadingHTTPServer(("127.0.0.1", int(puerto)), Manejador)


def pedir_clave() -> None:
    """Pide la clave sin eco si falta. Solo vive en este proceso: ni en pantalla ni en la terminal."""
    if os.environ.get("META_API_KEY", "").strip():
        return
    os.environ["META_API_KEY"] = getpass.getpass("META_API_KEY: ")
    if not os.environ["META_API_KEY"].strip():
        raise SystemExit("Falta META_API_KEY.")


def cargar_predictor(con_atisbo: bool):
    """Pesos fijos de modelo/ si se pide Atisbo; si no, un predictor que nunca especula."""
    raiz = Path(__file__).resolve().parents[1] / "modelo"
    epsilon = float(json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))["epsilon"])
    if not con_atisbo:
        return PredictorNulo(), epsilon
    from atisbo.modelo import Atisbo

    return Atisbo.cargar(raiz / "atisbo_seed0.pt"), epsilon


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Consola en vivo de ATISBO-SPEC.")
    ap.add_argument("--modo", choices=("api", "simulada"), default="simulada")
    ap.add_argument("--mundo", type=int, default=5000)
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--puerto", type=int, default=8792)
    ap.add_argument("--pausa", type=float, default=0.25)
    ap.add_argument("--sin-sam", action="store_true")
    ap.add_argument("--atisbo", action="store_true", help="enciende el GRU propio, que no es un modelo de Meta")
    ap.add_argument("--max-llm", type=int, default=400)
    ap.add_argument("--max-sam", type=int, default=500)
    ap.add_argument("--cache", type=str, default="demo/out/vivo_cache")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = _parse_args(argv)
    if a.modo == "api":
        pedir_clave()
    modelo, epsilon = cargar_predictor(a.atisbo)
    mundo = Mundo(seed=int(a.mundo), p_perturb=float(a.p))
    bus = Bus()
    if a.modo == "api":
        # Caché nueva en cada arranque: así ninguna respuesta sale de una corrida anterior.
        raiz = Path(a.cache) / str(int(time.time()))
        politica = PoliticaLLM(cliente=_ClienteVisto(ClienteMeta(modo="api", cache_dir=str(raiz / "llm")), bus), max_llm_calls=int(a.max_llm))
        cliente_sam = None if a.sin_sam else ClienteSam(modo="api", cache_dir=str(raiz / "sam"), max_red=int(a.max_sam))
        consola = Consola(modelo, politica, mundo, TAU, epsilon, a.modo, bus=bus, cliente_sam=cliente_sam, pausa=a.pausa, con_atisbo=a.atisbo)
    else:
        mascaras = None if a.sin_sam else mascaras_por_color
        consola = Consola(modelo, PoliticaSimulada(seed=int(a.mundo)), mundo, TAU, epsilon, a.modo, bus=bus, mascaras_de=mascaras, pausa=a.pausa, con_atisbo=a.atisbo)
    srv = crear_servidor(consola, a.puerto)
    print(f"Consola lista: http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
