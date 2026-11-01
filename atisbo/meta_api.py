"""Cliente de Meta Model API con caché registro/reproducción (PLIEGO §6)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.meta.ai/v1/chat/completions"
MODELO = "muse-spark-1.3"
REINTENTABLES = frozenset({429, 500, 503})
MAX_INTENTOS = 3


class CacheMiss(Exception):
    """Modo replay sin entrada en caché: nunca toca la red."""


class MetaAPIError(Exception):
    """Error de la API o de protocolo (clave, HTTP no reintentable, respuesta)."""


def huella(payload: dict) -> str:
    """sha256 del payload canónico (claves ordenadas)."""
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def _texto(cuerpo: dict) -> str | None:
    """Texto de choices[0].message.content o None si falta o no es texto."""
    if not isinstance(cuerpo, dict):
        return None
    choices = cuerpo.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    primero = choices[0]
    if not isinstance(primero, dict):
        return None
    mensaje = primero.get("message")
    if not isinstance(mensaje, dict):
        return None
    texto = mensaje.get("content")
    return texto if isinstance(texto, str) else None


def _entero(valor) -> int:
    if isinstance(valor, bool):
        return 0
    try:
        numero = int(valor)
    except (TypeError, ValueError):
        return 0
    return numero if numero >= 0 else 0


def _reintento_tras(cabeceras: dict):
    for clave, valor in cabeceras.items():
        if str(clave).lower() == "retry-after":
            try:
                espera = float(valor)
            except (TypeError, ValueError):
                return None
            return espera if espera >= 0 else None
    return None


class _TransporteUrllib:
    """Transporte real con urllib (biblioteca estándar)."""

    def __init__(self, timeout: float = 60.0) -> None:
        self._timeout = float(timeout)

    def post(self, url: str, cabeceras: dict, cuerpo: bytes):
        peticion = urllib.request.Request(url, data=cuerpo, headers=dict(cabeceras), method="POST")
        try:
            with urllib.request.urlopen(peticion, timeout=self._timeout) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers or {}), exc.read()


class ClienteMeta:
    """Chat Completions con caché en disco. replay nunca toca la red."""

    def __init__(self, modo: str, cache_dir="cache/llm", transporte=None, dormir=None, semilla: int = 0) -> None:
        if modo not in ("api", "replay"):
            raise ValueError("modo api|replay")
        self._modo = modo
        self._cache = Path(cache_dir)
        self._transporte = transporte if transporte is not None else _TransporteUrllib()
        self._dormir = dormir if dormir is not None else time.sleep
        self._rng = random.Random(semilla)
        self.ultima_desde_cache = False
        self.tokens_entrada = 0
        self.tokens_salida = 0

    def _ruta(self, payload: dict) -> Path:
        return self._cache / (huella(payload) + ".json")

    def tiene(self, payload: dict) -> bool:
        """True si la petición ya está en caché (sirve sin red)."""
        return self._ruta(payload).exists()

    def _sumar_tokens(self, uso: dict) -> None:
        if not isinstance(uso, dict):
            return
        self.tokens_entrada += _entero(uso.get("prompt_tokens", uso.get("input_tokens", 0)))
        self.tokens_salida += _entero(uso.get("completion_tokens", uso.get("output_tokens", 0)))

    def _guardar(self, ruta: Path, payload: dict, cuerpo_resp: dict, latencia_ms: float) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.parent / (ruta.name + ".tmp")
        tmp.write_text(json.dumps({"payload": payload, "cuerpo": cuerpo_resp, "latencia_ms": float(latencia_ms)}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, ruta)

    def completar(self, payload: dict):
        """Devuelve (texto, usage, latencia_ms). Sin forzar red: cada payload tiene su entrada."""
        ruta = self._ruta(payload)
        if ruta.exists():
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            cuerpo = datos["cuerpo"]
            uso = cuerpo.get("usage", {})
            if not isinstance(uso, dict):
                uso = {}
            self._sumar_tokens(uso)
            self.ultima_desde_cache = True
            return _texto(cuerpo) or "", uso, float(datos["latencia_ms"])
        if self._modo == "replay":
            raise CacheMiss("sin caché; replay no toca la red")
        return self._red(payload, ruta)

    def _red(self, payload: dict, ruta: Path):
        crudo = os.environ.get("META_API_KEY")
        if not crudo or not crudo.strip():
            raise MetaAPIError("falta META_API_KEY en el entorno")
        clave = crudo.strip()
        if any(ord(c) < 32 or ord(c) == 127 for c in clave):
            raise MetaAPIError("clave inválida (caracteres de control)")
        cuerpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        cabeceras = {"Authorization": "Bearer " + clave, "Content-Type": "application/json"}
        intento = 0
        latencia_ms = 0.0
        while True:
            try:
                t0 = time.perf_counter()
                status, cab_resp, bruto = self._transporte.post(BASE_URL, cabeceras, cuerpo)
                dt_ms = (time.perf_counter() - t0) * 1000.0
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if intento < MAX_INTENTOS - 1:
                    self._dormir(2 ** intento + self._rng.random())
                    intento += 1
                    continue
                raise MetaAPIError("red Meta: " + str(exc)) from exc
            if status == 200:
                latencia_ms = dt_ms
                break
            if status in REINTENTABLES and intento < MAX_INTENTOS - 1:
                espera = _reintento_tras(cab_resp)
                if espera is None:
                    espera = 2 ** intento + self._rng.random()
                self._dormir(espera)
                intento += 1
                continue
            raise MetaAPIError("HTTP " + str(status) + ": " + self._detalle(bruto))
        try:
            cuerpo_resp = json.loads(bruto.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise MetaAPIError("cuerpo no JSON") from exc
        uso = cuerpo_resp.get("usage", {})
        if not isinstance(uso, dict):
            uso = {}
        self._sumar_tokens(uso)
        texto = _texto(cuerpo_resp) or ""
        self._guardar(ruta, payload, cuerpo_resp, float(latencia_ms))
        self.ultima_desde_cache = False
        return texto, uso, float(latencia_ms)

    @staticmethod
    def _detalle(bruto: bytes) -> str:
        try:
            datos = json.loads(bruto.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return bruto[:200].decode("utf-8", errors="replace")
        if isinstance(datos, dict) and isinstance(datos.get("error"), dict):
            err = datos["error"]
            return str(err.get("message", err))[:300]
        return str(datos)[:300]
