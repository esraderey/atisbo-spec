"""Política LLM con muse-spark-1.3 y política simulada (PLIEGO §6). Sin red en tests."""
from __future__ import annotations

import json

import numpy as np

from atisbo.control import PredictorRuido
from atisbo.estado import Accion, vector
from atisbo.meta_api import MODELO

ACCIONES = ["ARRIBA", "ABAJO", "IZQ", "DER", "TOMAR", "SOLTAR", "ESPERAR"]

SYSTEM = (
    "Controlas un robot en cuadricula 12x12. Coordenadas: x crece a la derecha, y crece "
    "hacia abajo; (0,0) arriba a la izquierda. ARRIBA y-1, ABAJO y+1, IZQ x-1, DER x+1. "
    "Cubos: rojo (0,0), verde (11,0), azul (0,11). Objetivo: todos los objetos en su cubo "
    "del mismo color en 160 pasos como maximo. Acciones: ARRIBA, ABAJO, IZQ, DER, TOMAR, "
    "SOLTAR, ESPERAR. Moverse cambia una celda salvo borde o celda bloqueada. TOMAR solo "
    "sobre objeto libre sin carga. Libre es en_cubo false. El objeto que llevas sale de "
    "objetos y al soltarlo recibe un id nuevo. SOLTAR en su cubo lo guarda; si no, lo deja "
    "en la celda. Puede haber perturbaciones: objetos que se mueven o aparecen, bloqueo "
    "de una celda durante 5 pasos. carga es el tipo que llevas o null. Sin objetos libres "
    "ni carga, ESPERAR. Responde solo con la accion en JSON."
)


def parsear_accion(texto: str) -> Accion:
    """Parseo estricto de {"accion": "<A>"}; ValueError si no es exacto."""
    try:
        datos = json.loads(texto)
    except ValueError as exc:
        raise ValueError("no es JSON") from exc
    if not isinstance(datos, dict) or set(datos) != {"accion"}:
        raise ValueError("accion inválida")
    nombre = datos["accion"]
    if not isinstance(nombre, str) or nombre not in Accion.__members__:
        raise ValueError("accion inválida")
    return Accion[nombre]


def construir_payload(estado) -> dict:
    """system (reglas §1) + estado en JSON compacto + json_schema strict true."""
    objetos = [
        {"id": int(o[0]), "tipo": o[1], "x": int(o[2]), "y": int(o[3]), "en_cubo": bool(o[4])}
        for o in sorted(estado.objetos)
    ]
    usuario = {
        "bot": {"x": int(estado.bot[0]), "y": int(estado.bot[1])},
        "carga": estado.carga,
        "objetos": objetos,
        "cubos": {k: {"x": int(v[0]), "y": int(v[1])} for k, v in sorted(estado.cubos.items())},
        "bloqueo": {"x": int(estado.bloqueo[0]), "y": int(estado.bloqueo[1])} if estado.bloqueo else None,
        "paso": int(estado.paso),
    }
    return {
        "model": MODELO,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(usuario, separators=(",", ":"), ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "accion",
                "schema": {
                    "type": "object",
                    "properties": {"accion": {"type": "string", "enum": list(ACCIONES)}},
                    "required": ["accion"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        "reasoning_effort": "minimal",
    }


def construir_reintento(payload: dict, texto_invalido: str) -> dict:
    """Segundo payload distinto: avisa de la respuesta inválida; entrada propia en caché."""
    copia = dict(payload)
    mensajes = list(payload.get("messages", []))
    mensajes.append({
        "role": "user",
        "content": "Respuesta anterior inválida: " + str(texto_invalido)[:200] + ". Responde solo con {\"accion\": \"<ARRIBA|ABAJO|IZQ|DER|TOMAR|SOLTAR|ESPERAR>\"}.",
    })
    copia["messages"] = mensajes
    return copia


class PoliticaLLM:
    """Decide cada paso con el LLM; parseo estricto + 1 reintento, cae a ESPERAR."""

    def __init__(self, cliente, max_llm_calls: int = 1500) -> None:
        self.cliente = cliente
        self.max_llm_calls = int(max_llm_calls)
        self.modelo = MODELO
        self.llamadas_llm = 0
        self.peticiones = 0
        self.llamadas_red = 0
        self.llm_invalidas = 0
        self.truncada = False

    def _consume(self, payload: dict):
        texto, _uso, lat = self.cliente.completar(payload)
        self.peticiones += 1
        if not self.cliente.ultima_desde_cache:
            self.llamadas_red += 1
        return texto, float(lat)

    def decidir(self, estado):
        self.llamadas_llm += 1
        if self.truncada:
            return Accion.ESPERAR, 0.0
        payload = construir_payload(estado)
        if self.peticiones >= self.max_llm_calls:
            self.truncada = True
            return Accion.ESPERAR, 0.0
        texto, lat = self._consume(payload)
        try:
            accion = parsear_accion(texto)
        except ValueError:
            accion = None
        if accion is not None:
            return accion, lat
        reintento = construir_reintento(payload, texto)
        if self.peticiones >= self.max_llm_calls:
            self.truncada = True
            return Accion.ESPERAR, lat
        texto2, lat2 = self._consume(reintento)
        try:
            accion2 = parsear_accion(texto2)
        except ValueError:
            accion2 = None
        if accion2 is not None:
            return accion2, lat + lat2
        self.llm_invalidas += 1
        return Accion.ESPERAR, lat + lat2


class PoliticaSimulada:
    """PredictorRuido(0.1) como política; determinista por semilla. Sin red."""

    def __init__(self, seed: int = 0, p: float = 0.1) -> None:
        self.p = float(p)
        self._pred = PredictorRuido(self.p, seed=seed)
        self.llamadas_llm = 0
        self.peticiones = 0
        self.llamadas_red = 0
        self.llm_invalidas = 0
        self.truncada = False

    def decidir(self, estado):
        self.llamadas_llm += 1
        ventana = np.asarray([vector(estado)], dtype=np.float32)
        return self._pred.predecir(ventana).acciones[0], 0.0
