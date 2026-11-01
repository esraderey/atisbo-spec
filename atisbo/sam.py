"""Cliente SAM 3.1 con caché registro/reproducción, máscaras y seguidor (PLIEGO §2 y §6)."""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import os
import random
import tempfile
import time
import urllib.error
from pathlib import Path

import numpy as np
from PIL import Image
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format, video_segmentation_format

from atisbo.env.mundo import aplicar
from atisbo.env.render import a_mp4, render
from atisbo.env.rutina import elegir_accion
from atisbo.env.trazas import CUBOS_FIJOS
from atisbo.estado import Accion, EstadoSim, estado_desde_mascaras
from atisbo.meta_api import MAX_INTENTOS, REINTENTABLES, CacheMiss, MetaAPIError, _TransporteUrllib, _entero, _reintento_tras, huella

MODELO_SAM = "sam-3.1"
PROMPT = "colored square"
USD_POR_IMAGEN = 0.0025
USD_POR_FRAME = 0.0002
BASE_URL = "https://api.meta.ai/v1/responses"


class TopeSam(MetaAPIError):
    """Tope de peticiones SAM alcanzado."""


def _png_bytes(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _mp4_bytes(frames: list[np.ndarray]) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "ep.mp4"
        a_mp4([np.asarray(f, dtype=np.uint8) for f in frames], ruta)
        return ruta.read_bytes()


def _parse_sse(bruto: bytes) -> tuple[str, dict]:
    try:
        crudo = bruto.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MetaAPIError("cuerpo no UTF-8") from exc
    partes: list[str] = []
    uso: dict = {}
    completado = False
    for cruda in crudo.splitlines():
        linea = cruda.strip()
        if not linea.startswith("data:"):
            continue
        dato = linea[5:].strip()
        if dato == "[DONE]":
            continue
        try:
            obj = json.loads(dato)
        except ValueError as exc:
            raise MetaAPIError("línea data no JSON") from exc
        if not isinstance(obj, dict):
            continue
        tipo = obj.get("type")
        if tipo == "response.output_text.delta":
            delta = obj.get("delta", "")
            if isinstance(delta, str):
                partes.append(delta)
        elif tipo == "error" or tipo == "response.failed":
            raise MetaAPIError("evento de error SAM: " + str(tipo))
        elif tipo == "response.refusal.delta" or tipo == "response.refusal.done":
            raise MetaAPIError("evento de error SAM: " + str(tipo))
        elif tipo == "response.completed":
            completado = True
            resp = obj.get("response", {})
            if isinstance(resp, dict):
                u = resp.get("usage", {})
                uso = u if isinstance(u, dict) else {}
    if not completado:
        raise MetaAPIError("stream sin response.completed")
    return "".join(partes), uso


def _validar_texto(texto: str, medio: str) -> None:
    if medio == "imagen":
        fmt = image_segmentation_format()
    else:
        fmt = video_segmentation_format()
    parser = fmt.create_parser()
    list(parser.push(texto if isinstance(texto, str) else ""))
    res = parser.finish(CompletedOutcome()).result
    if any(d.severity == "error" for d in res.diagnostics):
        raise MetaAPIError("SAM inválido")
    if any(r.kind == "text" and r.text.strip() for r in res.records):
        raise MetaAPIError("SAM inválido: texto libre")


class ClienteSam:
    """Segmentación SAM 3.1 con caché en disco. replay nunca toca la red."""

    def __init__(self, modo: str, cache_dir="cache/sam", transporte=None, dormir=None, semilla: int = 0, max_red=None) -> None:
        if modo not in ("api", "replay"):
            raise ValueError("modo api|replay")
        self._modo = modo
        self._cache = Path(cache_dir)
        self._transporte = transporte if transporte is not None else _TransporteUrllib(timeout=300)
        self._dormir = dormir if dormir is not None else time.sleep
        self._rng = random.Random(semilla)
        self.ultima_desde_cache = False
        self.imagenes = 0
        self.frames_video = 0
        self.peticiones = 0
        self.llamadas_red = 0
        self.max_red = max_red
        self._latencias: list[float] = []

    def _ruta(self, payload: dict) -> Path:
        return self._cache / (huella(payload) + ".json")

    def _sumar_uso(self, uso: dict) -> None:
        if not isinstance(uso, dict):
            return
        self.imagenes += _entero(uso.get("images_processed", 0))
        self.frames_video += _entero(uso.get("video_frames_processed", 0))

    def coste_estimado_usd(self) -> float:
        return float(self.imagenes * USD_POR_IMAGEN + self.frames_video * USD_POR_FRAME)

    def latencia_ms_media(self) -> float:
        if not self._latencias:
            return 0.0
        return float(sum(self._latencias) / len(self._latencias))

    def segmentar(self, frames, prompt: str = PROMPT) -> str:
        arrs = [np.asarray(f, dtype=np.uint8) for f in frames]
        if not arrs:
            raise ValueError("sin frames")
        medio = "imagen" if len(arrs) == 1 else "video"
        payload = {"model": MODELO_SAM, "prompt": prompt, "medio": medio, "frames": hashlib.sha256(np.stack(arrs).tobytes()).hexdigest(), "n_frames": len(arrs), "metadata": {"include_confidence": "true"}}
        ruta = self._ruta(payload)
        self.peticiones += 1
        if ruta.exists():
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            uso = datos.get("uso", {})
            if not isinstance(uso, dict):
                uso = {}
            self._sumar_uso(uso)
            self.ultima_desde_cache = True
            self._latencias.append(float(datos["latencia_ms"]))
            texto = datos.get("texto", "")
            return texto if isinstance(texto, str) else ""
        if self._modo == "replay":
            raise CacheMiss("sin caché; replay no toca la red")
        if self.max_red is not None and self.llamadas_red >= self.max_red:
            raise TopeSam("tope de peticiones SAM")
        return self._red(payload, ruta, arrs, medio, prompt)

    def _red(self, payload: dict, ruta: Path, arrs: list[np.ndarray], medio: str, prompt: str) -> str:
        crudo = os.environ.get("META_API_KEY")
        if not crudo or not crudo.strip():
            raise MetaAPIError("falta META_API_KEY en el entorno")
        clave = crudo.strip()
        if any(ord(c) < 32 or ord(c) == 127 for c in clave):
            raise MetaAPIError("clave inválida (caracteres de control)")
        if medio == "imagen":
            b64 = base64.b64encode(_png_bytes(arrs[0])).decode("ascii")
            medio_obj = {"type": "input_image", "image_url": "data:image/png;base64," + b64}
        else:
            b64 = base64.b64encode(_mp4_bytes(arrs)).decode("ascii")
            medio_obj = {"type": "input_video", "video_url": "data:video/mp4;base64," + b64}
        cuerpo = {"model": MODELO_SAM, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, medio_obj]}], "stream": True, "metadata": {"include_confidence": "true"}}
        crudo_cuerpo = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        cabeceras = {"Authorization": "Bearer " + clave, "Content-Type": "application/json", "Accept": "text/event-stream"}
        intento = 0
        latencia_ms = 0.0
        bruto = b""
        texto = ""
        uso: dict = {}
        while True:
            try:
                t0 = time.perf_counter()
                status, cab_resp, bruto = self._transporte.post(BASE_URL, cabeceras, crudo_cuerpo)
                dt_ms = (time.perf_counter() - t0) * 1000.0
            except (urllib.error.URLError, TimeoutError) as exc:
                if intento < MAX_INTENTOS - 1:
                    self._dormir(2 ** intento + self._rng.random())
                    intento += 1
                    continue
                raise MetaAPIError("red SAM: " + str(exc)) from exc
            if status == 200:
                latencia_ms = dt_ms
                try:
                    texto, uso = _parse_sse(bruto)
                except MetaAPIError:
                    if intento < MAX_INTENTOS - 1:
                        self._dormir(2 ** intento + self._rng.random())
                        intento += 1
                        continue
                    raise
                break
            if (status in REINTENTABLES or status in (502, 504)) and intento < MAX_INTENTOS - 1:
                espera = _reintento_tras(cab_resp)
                if espera is None:
                    espera = 2 ** intento + self._rng.random()
                self._dormir(espera)
                intento += 1
                continue
            raise MetaAPIError("HTTP " + str(status) + ": " + bruto[:200].decode("utf-8", errors="replace"))
        _validar_texto(texto, medio)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.parent / (ruta.name + ".tmp")
        tmp.write_text(json.dumps({"payload": payload, "texto": texto, "uso": uso, "latencia_ms": float(latencia_ms)}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, ruta)
        self._sumar_uso(uso)
        self.llamadas_red += 1
        self.ultima_desde_cache = False
        self._latencias.append(float(latencia_ms))
        return texto


def mascaras_por_frame(texto: str, medio: str) -> dict[int, dict[int, np.ndarray]]:
    """Texto SAM → {frame_index: {object_id: máscara bool 384×384}}."""
    if medio == "imagen":
        fmt = image_segmentation_format()
    elif medio == "video":
        fmt = video_segmentation_format()
    else:
        raise ValueError("medio imagen|video")
    parser = fmt.create_parser()
    list(parser.push(texto if isinstance(texto, str) else ""))
    registros = parser.finish(CompletedOutcome()).result.records
    salida: dict[int, dict[int, np.ndarray]] = {}
    for rec in registros:
        if rec.kind != "mask":
            continue
        oid = int(rec.object_id)
        left = rec.bounds.left
        top = rec.bounds.top
        right = rec.bounds.right
        bottom = rec.bounds.bottom
        fi = 0 if medio == "imagen" else rec.frame.frame_index
        arr = (np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(rec.mask.height, rec.mask.width) != 0)
        lienzo = np.zeros((384, 384), dtype=bool)
        lienzo[top:bottom, left:right] = arr
        salida.setdefault(fi, {})[oid] = lienzo
    if medio == "imagen" and 0 not in salida:
        salida[0] = {}
    return salida


class SeguidorEstado:
    """Memoria de oclusiones e ids sobre EstadoSam (PLIEGO §2)."""

    def __init__(self) -> None:
        self.ciego = False
        self._prev = None
        self._faltas: dict[int, int] = {}

    def actualizar(self, visto: EstadoSim, accion=None) -> EstadoSim:
        if self._prev is None:
            self.ciego = visto.bot is None
            res = EstadoSim(bot=(0, 0) if visto.bot is None else visto.bot, carga=None if visto.bot is None else visto.carga, objetos=tuple(sorted(visto.objetos, key=lambda o: o[0])), cubos=dict(CUBOS_FIJOS), bloqueo=visto.bloqueo, paso=0)
            self._prev = res
            return res
        prev = self._prev
        nxt = max([o[0] for o in prev.objetos] + [o[0] for o in visto.objetos] + [0]) + 1
        if visto.bot is None:
            self.ciego = True
            previsto = aplicar(prev, accion) if accion is not None else prev
            bot = previsto.bot
            carga = previsto.carga
            base = list(previsto.objetos)
        else:
            self.ciego = False
            bot, carga = visto.bot, visto.carga
            base = list(prev.objetos)
            if visto.carga is not None and prev.carga != visto.carga:
                cands = [o for o in base if o[1] == visto.carga and not o[4] and (o[2], o[3]) == bot]
                if cands:
                    q = min(cands, key=lambda o: o[0])[0]
                    base = [o for o in base if o[0] != q]
                    self._faltas.pop(q, None)
            if accion is Accion.TOMAR and prev.carga is None and visto.carga is None:
                quitar = {o[0] for o in base if not o[4] and (o[2], o[3]) == bot}
                if quitar:
                    base = [o for o in base if o[0] not in quitar]
                    for q in quitar:
                        self._faltas.pop(q, None)
            vistos_set = set((tt, xx, yy) for _, tt, xx, yy, _ in visto.objetos)
            filtrada: list = []
            for o in base:
                oid, t, x, y, _e = o
                if (t, x, y) in vistos_set or (x, y) == bot:
                    self._faltas.pop(oid, None)
                    filtrada.append(o)
                else:
                    f = self._faltas.get(oid, 0)
                    if f >= 1:
                        self._faltas.pop(oid, None)
                    else:
                        self._faltas[oid] = f + 1
                        filtrada.append(o)
            base = filtrada
            if prev.carga is not None and prev.carga != visto.carga:
                base.append((nxt, prev.carga, bot[0], bot[1], CUBOS_FIJOS.get(prev.carga) == bot))
                nxt += 1
        claves = set((t, x, y) for _, t, x, y, _ in base)
        for _, t, x, y, _ in sorted(visto.objetos, key=lambda o: o[0]):
            if (t, x, y) not in claves:
                base.append((nxt, t, x, y, CUBOS_FIJOS.get(t) == (x, y)))
                claves.add((t, x, y))
                nxt += 1
        base.sort(key=lambda o: o[0])
        res = EstadoSim(bot=bot, carga=carga, objetos=tuple(base), cubos=dict(CUBOS_FIJOS), bloqueo=visto.bloqueo, paso=prev.paso + 1)
        self._prev = res
        return res


class MundoSam:
    """El lazo ve solo lo que reconstruye SAM; la dinámica es la del Mundo real."""

    def __init__(self, mundo, mascaras_de) -> None:
        self._mundo = mundo
        self._mascaras_de = mascaras_de
        self._real = None
        self._seg = SeguidorEstado()
        self.ciego = False
        self.frames = 0
        self.pasos_ciegos = 0
        self._ok_bot = 0
        self._ok_obj = 0
        self._ok_todo = 0
        self.rutinas_reales: list[str] = []

    @property
    def exito(self) -> bool:
        return self._mundo.exito

    def reiniciar(self):
        self._real = self._mundo.reiniciar()
        self._seg = SeguidorEstado()
        self.ciego = False
        self.frames = 0
        self.pasos_ciegos = 0
        self._ok_bot = 0
        self._ok_obj = 0
        self._ok_todo = 0
        self.rutinas_reales = []
        return self._percibir()

    def avanzar(self, s, a):
        self.rutinas_reales.append(elegir_accion(self._real).name)
        self._real = self._mundo.avanzar(self._real, a)
        return self._percibir(a)

    def _percibir(self, accion=None):
        frame = render(self._real)
        mascaras = self._mascaras_de(frame)
        visto = estado_desde_mascaras(mascaras, frame)
        perc = self._seg.actualizar(visto, accion)
        self.ciego = self._seg.ciego
        self.frames += 1
        if self.ciego:
            self.pasos_ciegos += 1
        real = self._real
        got_obj = sorted((t, x, y, bool(e)) for _, t, x, y, e in perc.objetos)
        exp_obj = sorted((t, x, y, bool(e)) for _, t, x, y, e in real.objetos)
        if perc.bot == real.bot:
            self._ok_bot += 1
        if got_obj == exp_obj:
            self._ok_obj += 1
        if perc.bot == real.bot and perc.carga == real.carga and got_obj == exp_obj and perc.bloqueo == real.bloqueo:
            self._ok_todo += 1
        return perc

    def coincidencias(self) -> dict:
        if self.frames == 0:
            return {"bot": 0.0, "objetos": 0.0, "todo": 0.0}
        return {"bot": float(self._ok_bot / self.frames), "objetos": float(self._ok_obj / self.frames), "todo": float(self._ok_todo / self.frames)}


class PredictorCiego:
    """Fuerza LLM cuando SAM no ve al bot, sin tocar lazo.py."""

    def __init__(self, predictor, mundo_sam) -> None:
        self._predictor = predictor
        self._mundo_sam = mundo_sam

    def predecir(self, ventana):
        pred = self._predictor.predecir(ventana)
        if self._mundo_sam.ciego:
            # por debajo de cualquier τ
            return dataclasses.replace(pred, confianza=-1.0)
        return pred
