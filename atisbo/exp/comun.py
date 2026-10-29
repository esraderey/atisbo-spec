"""Utilidades comunes de experimentos (PLIEGO §8): trazas, bootstrap, results, commit, validador."""

from __future__ import annotations

import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atisbo.env.trazas import generar
from atisbo.estado import vector

K = 8
_SOPORTADAS = {
    "type", "const", "enum", "required", "properties",
    "additionalProperties", "items", "minItems", "maxItems",
    "minimum", "maximum", "allOf", "if", "then", "else",
}
_IGNORADAS = {"$schema", "$id", "title", "description"}
_TIPOS = {"object", "array", "string", "integer", "number", "boolean", "null"}


class SchemaError(Exception):
    """El propio esquema es inválido (palabra no soportada, tipo desconocido)."""


def ventana(episodio, idx: int, k: int = K) -> np.ndarray:
    """Ventana de k vectores s_{t-k+1..t} con relleno del primero."""
    n = len(episodio)
    vecs = []
    for j in range(k):
        ii = idx - (k - 1 - j)
        if ii < 0:
            ii = 0
        if ii >= n:
            ii = n - 1
        vecs.append(vector(episodio[ii][0]))
    return np.stack(vecs).astype(np.float32)


def construir_muestras(episodios, k: int = K):
    """Muestras §3: X (N,k,D), A (N,3) int, S (N,3,D) float32. Últimos 3 pasos sin muestra."""
    xs, aa, ss = [], [], []
    d = None
    for ep in episodios:
        n = len(ep)
        if n < 4:
            continue
        if d is None:
            d = len(vector(ep[0][0]))
        for t in range(n - 3):
            xs.append(ventana(ep, t, k))
            aa.append([int(ep[t][1].value), int(ep[t + 1][1].value), int(ep[t + 2][1].value)])
            ss.append([vector(ep[t + 1][0]), vector(ep[t + 2][0]), vector(ep[t + 3][0])])
    if not xs:
        raise ValueError("sin muestras: episodios demasiado cortos")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(aa, dtype=np.int64),
        np.stack(ss).astype(np.float32),
    )


def split_episodios(episodios, seed: int, ratio: float = 0.8):
    """Split por episodio con semilla fija."""
    idx = list(range(len(episodios)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    ntr = int(ratio * len(episodios))
    return [episodios[i] for i in idx[:ntr]], [episodios[i] for i in idx[ntr:]]


def bootstrap(valores, n_remuestreos: int = 1000, seed: int = 0) -> dict:
    """Bootstrap no paramétrico de la media; reproducible con semilla fija."""
    vals = [float(v) for v in valores]
    n = len(vals)
    if n == 0:
        raise ValueError("bootstrap sin valores")
    media = float(sum(vals) / n)
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=np.float64)
    medias = np.empty(n_remuestreos, dtype=np.float64)
    for b in range(n_remuestreos):
        medias[b] = arr[rng.randint(0, n, size=n)].mean()
    lo = float(np.percentile(medias, 2.5))
    hi = float(np.percentile(medias, 97.5))
    return {"media": media, "ic95": [lo, hi], "n": n, "n_remuestreos": int(n_remuestreos)}


def commit_hash() -> str:
    """Hash del commit; '+sucio' si hay cambios sin commitear en atisbo/ o pyproject.toml."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        h = r.stdout.strip()
        if r.returncode != 0 or not h:
            return "sin-git"
        try:
            s = subprocess.run(
                ["git", "status", "--porcelain", "--", "atisbo", "pyproject.toml"],
                capture_output=True, text=True, timeout=10,
            )
            if s.returncode != 0:
                return h + "+sucio"
            if s.stdout.strip():
                return h + "+sucio"
        except (OSError, subprocess.SubprocessError):
            return h + "+sucio"
        return h
    except (OSError, subprocess.SubprocessError):
        return "sin-git"


def fecha_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def escribir_json(ruta, datos) -> None:
    p = Path(ruta)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(datos, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def generar_trazas(seed: int, n_episodios: int, p_perturb: float):
    return generar(seed, n_episodios, p_perturb)


def _igual_json(a, b) -> bool:
    """Igualdad JSON estricta: bool no se confunde con 1/0; números por valor."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return False
        return fa == fb
    if isinstance(a, str) or isinstance(b, str):
        return type(a) is type(b) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_igual_json(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_igual_json(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _comprobar_esquema(esquema, ruta: str = "$") -> None:
    """Comprueba todo el esquema antes de validar; errores propios -> SchemaError."""
    if isinstance(esquema, bool):
        return
    if not isinstance(esquema, dict):
        raise SchemaError(f"{ruta}: esquema inválido")
    for clave in esquema:
        if clave not in _SOPORTADAS and clave not in _IGNORADAS:
            raise SchemaError(f"{ruta}: palabra no soportada {clave!r}")
    tipo = esquema.get("type")
    if tipo is not None:
        if isinstance(tipo, str):
            tipos = [tipo]
        elif isinstance(tipo, list) and all(isinstance(t, str) for t in tipo):
            tipos = list(tipo)
        else:
            raise SchemaError(f"{ruta}: type inválido {tipo!r}")
        for t in tipos:
            if t not in _TIPOS:
                raise SchemaError(f"{ruta}: tipo desconocido {t!r}")
    if "allOf" in esquema:
        subs = esquema["allOf"]
        if not isinstance(subs, list):
            raise SchemaError(f"{ruta}: allOf debe ser lista")
        for i, sub in enumerate(subs):
            _comprobar_esquema(sub, f"{ruta}.allOf[{i}]")
    for clave in ("if", "then", "else"):
        if clave in esquema:
            _comprobar_esquema(esquema[clave], f"{ruta}.{clave}")
    if "properties" in esquema:
        props = esquema["properties"]
        if not isinstance(props, dict):
            raise SchemaError(f"{ruta}: properties debe ser objeto")
        for k, sub in props.items():
            _comprobar_esquema(sub, f"{ruta}.properties.{k}")
    if "items" in esquema:
        _comprobar_esquema(esquema["items"], f"{ruta}.items")
    if "additionalProperties" in esquema:
        adic = esquema["additionalProperties"]
        if isinstance(adic, dict):
            _comprobar_esquema(adic, f"{ruta}.additionalProperties")
        elif adic not in (True, False):
            raise SchemaError(f"{ruta}: additionalProperties inválido")


def validar(datos, esquema) -> None:
    """Validador JSON Schema draft 7 mínimo, sin dependencias."""
    _comprobar_esquema(esquema, "$")
    _validar(datos, esquema, "$")


def _validar(nodo, esquema, ruta: str) -> None:
    if isinstance(esquema, bool):
        if esquema is True:
            return
        raise ValueError(f"{ruta}: esquema false")
    if not isinstance(esquema, dict):
        raise SchemaError(f"{ruta}: esquema inválido")
    for clave in esquema:
        if clave not in _SOPORTADAS and clave not in _IGNORADAS:
            raise SchemaError(f"{ruta}: palabra no soportada {clave!r}")
    if isinstance(nodo, float) and not math.isfinite(nodo):
        raise ValueError(f"{ruta}: NaN o infinito no es JSON válido")
    if "allOf" in esquema:
        subs = esquema["allOf"]
        if not isinstance(subs, list):
            raise SchemaError(f"{ruta}: allOf debe ser lista")
        for i, sub in enumerate(subs):
            _validar(nodo, sub, f"{ruta}.allOf[{i}]")
    if "if" in esquema:
        try:
            _validar(nodo, esquema["if"], ruta)
            cond = True
        except ValueError:
            cond = False
        if cond and "then" in esquema:
            _validar(nodo, esquema["then"], ruta)
        if not cond and "else" in esquema:
            _validar(nodo, esquema["else"], ruta)
    if "const" in esquema and not _igual_json(nodo, esquema["const"]):
        raise ValueError(f"{ruta}: debe ser {esquema['const']!r}")
    if "enum" in esquema:
        opts = esquema["enum"]
        if not isinstance(opts, list) or not any(_igual_json(nodo, o) for o in opts):
            raise ValueError(f"{ruta}: {nodo!r} no está en {esquema['enum']!r}")
    tipo = esquema.get("type")
    if tipo is not None:
        tipos = [tipo] if isinstance(tipo, str) else list(tipo)
        if not any(_es_tipo(nodo, t) for t in tipos):
            raise ValueError(f"{ruta}: tipo {type(nodo).__name__} no es {tipos}")
    if isinstance(nodo, dict):
        for req in esquema.get("required", []):
            if req not in nodo:
                raise ValueError(f"{ruta}: falta {req!r}")
        props = esquema.get("properties", {})
        for clave, sub in props.items():
            if clave in nodo:
                _validar(nodo[clave], sub, f"{ruta}.{clave}")
        if "additionalProperties" in esquema:
            adic = esquema["additionalProperties"]
            for clave in nodo:
                if clave not in props:
                    if adic is False:
                        raise ValueError(f"{ruta}: propiedad adicional {clave!r}")
                    elif isinstance(adic, dict):
                        _validar(nodo[clave], adic, f"{ruta}.{clave}")
                    elif adic is True:
                        pass
                    else:
                        raise SchemaError(f"{ruta}: additionalProperties inválido")
    if isinstance(nodo, list):
        if "minItems" in esquema and len(nodo) < esquema["minItems"]:
            raise ValueError(f"{ruta}: mínimo {esquema['minItems']} elementos")
        if "maxItems" in esquema and len(nodo) > esquema["maxItems"]:
            raise ValueError(f"{ruta}: máximo {esquema['maxItems']} elementos")
        if "items" in esquema:
            sub = esquema["items"]
            for i, elem in enumerate(nodo):
                _validar(elem, sub, f"{ruta}[{i}]")
    if isinstance(nodo, (int, float)) and not isinstance(nodo, bool):
        if "minimum" in esquema and float(nodo) < esquema["minimum"]:
            raise ValueError(f"{ruta}: {nodo} < mínimo {esquema['minimum']}")
        if "maximum" in esquema and float(nodo) > esquema["maximum"]:
            raise ValueError(f"{ruta}: {nodo} > máximo {esquema['maximum']}")


def _es_tipo(nodo, tipo: str) -> bool:
    if tipo == "object":
        return isinstance(nodo, dict)
    if tipo == "array":
        return isinstance(nodo, list)
    if tipo == "string":
        return isinstance(nodo, str)
    if tipo == "integer":
        return isinstance(nodo, int) and not isinstance(nodo, bool)
    if tipo == "number":
        return isinstance(nodo, (int, float)) and not isinstance(nodo, bool)
    if tipo == "boolean":
        return isinstance(nodo, bool)
    if tipo == "null":
        return nodo is None
    raise SchemaError(f"tipo desconocido: {tipo}")
