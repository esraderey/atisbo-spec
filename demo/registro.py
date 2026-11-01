"""Registro paso a paso del lazo E2 (O7 parte 1). Sin red en tests salvo replay con cache."""

from __future__ import annotations

import argparse

from atisbo.estado import EstadoSim
from atisbo.lazo import correr_episodio


def estado_a_dict(estado: EstadoSim) -> dict:
    """EstadoSim a dict JSON."""
    return {
        "bot": [int(estado.bot[0]), int(estado.bot[1])],
        "carga": None if estado.carga is None else str(estado.carga),
        "objetos": [
            [int(o[0]), str(o[1]), int(o[2]), int(o[3]), bool(o[4])]
            for o in estado.objetos
        ],
        "cubos": {str(k): [int(v[0]), int(v[1])] for k, v in estado.cubos.items()},
        "bloqueo": None if estado.bloqueo is None else [int(estado.bloqueo[0]), int(estado.bloqueo[1])],
        "paso": int(estado.paso),
    }


def estado_desde_dict(d: dict) -> EstadoSim:
    """Dict JSON a EstadoSim (inversa exacta de estado_a_dict)."""
    bot = (int(d["bot"][0]), int(d["bot"][1]))
    carga = None if d["carga"] is None else str(d["carga"])
    objetos = tuple(
        (int(o[0]), str(o[1]), int(o[2]), int(o[3]), bool(o[4]))
        for o in d["objetos"]
    )
    cubos = {str(k): (int(v[0]), int(v[1])) for k, v in d["cubos"].items()}
    bloqueo = None if d["bloqueo"] is None else (int(d["bloqueo"][0]), int(d["bloqueo"][1]))
    return EstadoSim(bot=bot, carga=carga, objetos=objetos, cubos=cubos, bloqueo=bloqueo, paso=int(d["paso"]))


class _PredictorGrabador:
    """Envuelve predecir; guarda nombres de pred.acciones y reinicia la posición del bloque."""

    def __init__(self, interno) -> None:
        self._interno = interno
        self.ultimas = []
        self.pos = 0

    def predecir(self, ventana):
        pred = self._interno.predecir(ventana)
        self.ultimas = [str(a.name) for a in pred.acciones]
        self.pos = 0
        return pred

    def __getattr__(self, nombre):
        return getattr(self.__dict__["_interno"], nombre)


class _MundoGrabador:
    """Envuelve reiniciar/avanzar y guarda cada estado devuelto."""

    def __init__(self, interno) -> None:
        self._interno = interno
        self.estados = []

    def reiniciar(self):
        s = self._interno.reiniciar()
        self.estados.append(s)
        return s

    def avanzar(self, s, a):
        s2 = self._interno.avanzar(s, a)
        self.estados.append(s2)
        return s2

    def __getattr__(self, nombre):
        return getattr(self.__dict__["_interno"], nombre)


class _RegistroConPredichas(list):
    """Lista que anota en cada entrada solo lo que queda del bloque y avanza la posición."""

    def __init__(self, predictor) -> None:
        super().__init__()
        self._predictor = predictor

    def append(self, entrada) -> None:
        copia = dict(entrada)
        ultimas = list(self._predictor.ultimas)
        pos = int(self._predictor.pos)
        copia["_predichas"] = ultimas[pos:]
        self._predictor.pos = pos + 1
        super().append(copia)


def capturar(predictor, politica, mundo, tau, epsilon, horizonte: int = 3) -> dict:
    """Corre un episodio grabando estados y, por paso, solo las predichas restantes del bloque.

    El lazo predice una vez por bloque: el paso j del bloque lleva pred.acciones[j:].
    Un paso llm tras sorpresa lleva igualmente lo que quedaba sin ejecutar.
    """
    pred_g = _PredictorGrabador(predictor)
    mundo_g = _MundoGrabador(mundo)
    registro = _RegistroConPredichas(pred_g)
    res = correr_episodio(
        pred_g, politica, mundo_g,
        tau=float(tau), epsilon=float(epsilon),
        registro=registro, horizonte=int(horizonte),
    )
    pasos = []
    for i, e in enumerate(registro):
        estado = mundo_g.estados[i]
        pasos.append(
            {
                "estado": estado_a_dict(estado),
                "fuente": str(e["fuente"]),
                "accion": str(e["accion"]),
                "conf": float(e["conf"]),
                "sorpresa": float(e["sorpresa"]),
                "predichas": list(e.get("_predichas", [])),
            }
        )
    ultimo = mundo_g.estados[-1] if mundo_g.estados else None
    return {
        "exito": bool(res["exito"]),
        "llamadas_llm": int(res["llamadas_llm"]),
        "pasos": pasos,
        "estado_final": estado_a_dict(ultimo) if ultimo is not None else None,
    }


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Registra un episodio E2 con replay.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodio", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--cache", type=str, default="cache/llm")
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> str:
    """Reproduce E2 (entreno + epsilon) y registra el episodio pedido con replay."""
    from atisbo.exp.comun import escribir_json, generar_trazas, split_episodios
    from atisbo.exp.e2 import semilla_mundo
    from atisbo.lazo import Mundo
    from atisbo.meta_api import ClienteMeta
    from atisbo.modelo import Atisbo
    from atisbo.politica_llm import PoliticaLLM
    from atisbo.sorpresa import calibrar_epsilon

    a = _parse_args(argv)
    s = int(a.seed)
    objetivo = int(a.episodio)
    if objetivo < 0:
        raise ValueError("episodio debe ser >= 0")
    tau = float(a.tau)
    eps_tr = generar_trazas(s, 200, 0.02)
    tr, _ = split_episodios(eps_tr, s)
    modelo = Atisbo.entrenar(tr, seed=s, epocas=10)
    eps_cal = generar_trazas(s, 200, 0.0)
    _, te0 = split_episodios(eps_cal, s)
    epsilon = float(calibrar_epsilon(modelo, te0))
    mundo = Mundo(seed=semilla_mundo(s), p_perturb=0.02)
    cliente = ClienteMeta(modo="replay", cache_dir=str(a.cache))
    politica = PoliticaLLM(cliente=cliente)
    cap = None
    for _ in range(objetivo + 1):
        cap = capturar(modelo, politica, mundo, tau=tau, epsilon=epsilon)
    datos = {
        "seed": s,
        "episodio": objetivo,
        "tau": tau,
        "epsilon": float(epsilon),
        "exito": bool(cap["exito"]),
        "llamadas_llm": int(cap["llamadas_llm"]),
        "pasos": list(cap["pasos"]),
        "estado_final": dict(cap["estado_final"]),
    }
    escribir_json(str(a.out), datos)
    return str(a.out)


if __name__ == "__main__":
    main()
