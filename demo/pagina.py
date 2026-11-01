"""Datos de index.html: los episodios de una semilla de E2, por replay y sin red."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atisbo.env.mundo import aplicar
from atisbo.estado import Accion
from demo.registro import capturar, estado_desde_dict


def perturbaciones(pasos, final) -> list:
    """Pasos donde un objeto cambia o aparece un bloqueo sin que lo explique la acción anterior."""
    estados = [estado_desde_dict(p["estado"]) for p in pasos] + [estado_desde_dict(final)]
    out = []
    for i, p in enumerate(pasos):
        previsto, real = aplicar(estados[i], Accion[p["accion"]]), estados[i + 1]
        if real.objetos != previsto.objetos or real.bloqueo not in (None, previsto.bloqueo):
            out.append(i + 1)
    return out


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Genera results/pagina.json con replay.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--cache", type=str, default="cache/llm")
    ap.add_argument("--out", type=str, default="results/pagina.json")
    return ap.parse_args(argv)


def main(argv=None) -> str:
    from atisbo.exp.comun import generar_trazas, split_episodios
    from atisbo.exp.e2 import semilla_mundo
    from atisbo.lazo import Mundo
    from atisbo.meta_api import ClienteMeta
    from atisbo.modelo import Atisbo
    from atisbo.politica_llm import PoliticaLLM
    from atisbo.sorpresa import calibrar_epsilon

    a = _parse_args(argv)
    s = int(a.seed)
    tr, _ = split_episodios(generar_trazas(s, 200, 0.02), s)
    modelo = Atisbo.entrenar(tr, seed=s, epocas=10)
    _, te0 = split_episodios(generar_trazas(s, 200, 0.0), s)
    epsilon = float(calibrar_epsilon(modelo, te0))
    mundo = Mundo(seed=semilla_mundo(s), p_perturb=0.02)
    politica = PoliticaLLM(cliente=ClienteMeta(modo="replay", cache_dir=str(a.cache)))
    episodios = []
    for k in range(int(a.n)):
        cap = capturar(modelo, politica, mundo, tau=float(a.tau), epsilon=epsilon)
        pasos = [{"estado": p["estado"], "fuente": p["fuente"], "accion": p["accion"], "conf": round(p["conf"], 3), "predichas": p["predichas"]} for p in cap["pasos"]]
        episodios.append({"episodio": k, "exito": cap["exito"], "llamadas_llm": cap["llamadas_llm"], "perturbaciones": perturbaciones(cap["pasos"], cap["estado_final"]), "estado_final": cap["estado_final"], "pasos": pasos})
    datos = {"seed": s, "tau": float(a.tau), "epsilon": epsilon, "episodios": episodios}
    Path(a.out).write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    return str(a.out)


if __name__ == "__main__":
    main()
