"""Entrena el GRU publicado y escribe modelo/: pesos fijos, datos de entrenamiento y manifiesto. Sin red."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from atisbo.exp.comun import construir_muestras, generar_trazas, split_episodios
from atisbo.lazo import K
from atisbo.modelo import Atisbo
from atisbo.sorpresa import calibrar_epsilon

RAIZ = Path(__file__).resolve().parents[1] / "modelo"
SEED, EPISODIOS, P_PERTURB, EPOCAS = 0, 200, 0.02, 10


def entrenar_publicado():
    """Mismo entrenamiento que la semilla 0 de E2: devuelve el modelo y su umbral de sorpresa."""
    tr, _ = split_episodios(generar_trazas(SEED, EPISODIOS, P_PERTURB), SEED)
    modelo = Atisbo.entrenar(tr, seed=SEED, epocas=EPOCAS)
    _, te0 = split_episodios(generar_trazas(SEED, EPISODIOS, 0.0), SEED)
    return modelo, float(calibrar_epsilon(modelo, te0))


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main() -> None:
    modelo, epsilon = entrenar_publicado()
    RAIZ.mkdir(exist_ok=True)
    pesos, datos = RAIZ / "atisbo_seed0.pt", RAIZ / "datos_entrenamiento.npz"
    modelo.guardar(pesos)
    tr, _ = split_episodios(generar_trazas(SEED, EPISODIOS, P_PERTURB), SEED)
    X, A, S = construir_muestras(tr, K)
    np.savez_compressed(datos, X=X, A=A, S=S)
    manifiesto = {
        "semilla": SEED, "episodios_generados": EPISODIOS, "episodios_entrenamiento": len(tr), "p_perturb": P_PERTURB, "epocas": EPOCAS,
        "muestras": int(X.shape[0]), "parametros": modelo.n_parametros(), "epsilon": epsilon,
        "pesos_sha256": _sha(pesos), "datos_sha256": _sha(datos),
    }
    (RAIZ / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifiesto))


if __name__ == "__main__":
    main()
