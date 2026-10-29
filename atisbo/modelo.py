"""Atisbo (PLIEGO §3): GRU(128) + cabezas lineales, ≤1M parámetros, CPU determinista."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from atisbo.control import Prediccion
from atisbo.estado import Accion
from atisbo.exp.comun import construir_muestras

HIDDEN = 128
K = 8
N_ACCIONES = 7


class _Red(nn.Module):
    def __init__(self, d_in: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.cab_acc = nn.ModuleList([nn.Linear(hidden, N_ACCIONES) for _ in range(3)])
        self.cab_est = nn.ModuleList([nn.Linear(hidden, d_in) for _ in range(3)])

    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        logits = torch.stack([c(h) for c in self.cab_acc], dim=1)
        estados = torch.stack([c(h) for c in self.cab_est], dim=1)
        return logits, estados


class Atisbo:
    def __init__(self, d_in: int, hidden: int = HIDDEN) -> None:
        self._d_in = int(d_in)
        self._hidden = int(hidden)
        self._red = _Red(self._d_in, self._hidden)

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self._red.parameters())

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self._red.state_dict().items()}

    def predecir(self, ventana) -> Prediccion:
        self._red.eval()
        x = np.asarray(ventana, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self._d_in:
            raise ValueError(f"ventana debe ser (k,{self._d_in}), llego {x.shape}")
        with torch.no_grad():
            xt = torch.from_numpy(x[np.newaxis, :, :])
            logits, estados = self._red(xt)
            probs = torch.softmax(logits[0], dim=-1).numpy()
            idx = [int(np.argmax(probs[j])) for j in range(3)]
            conf = float(np.prod([float(probs[j, idx[j]]) for j in range(3)]))
            est = estados[0].numpy().astype(np.float32)
        return Prediccion(
            acciones=tuple(Accion(i) for i in idx),
            estados=est,
            confianza=conf,
        )

    def guardar(self, ruta) -> None:
        p = Path(ruta)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"d_in": self._d_in, "hidden": self._hidden, "sd": self._red.state_dict()},
            str(p),
        )

    @classmethod
    def cargar(cls, ruta) -> "Atisbo":
        blob = torch.load(str(ruta), map_location="cpu", weights_only=True)
        m = cls(int(blob["d_in"]), int(blob.get("hidden", HIDDEN)))
        m._red.load_state_dict(blob["sd"])
        m._red.eval()
        return m

    @classmethod
    def entrenar(cls, episodios, seed: int, epocas: int) -> "Atisbo":
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        X, A, S = construir_muestras(episodios, K)
        d_in = int(X.shape[2])
        m = cls(d_in)
        n = int(X.shape[0])
        Xt = torch.from_numpy(X)
        At = torch.from_numpy(A)
        St = torch.from_numpy(S)
        opt = torch.optim.Adam(m._red.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        m._red.train()
        bs = 256
        g = torch.Generator()
        g.manual_seed(int(seed))
        for _ in range(int(epocas)):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                xb, ab, sb = Xt[b], At[b], St[b]
                opt.zero_grad()
                logits, est = m._red(xb)
                loss = 0.0
                for j in range(3):
                    loss = loss + ce(logits[:, j, :], ab[:, j])
                loss = loss + 0.1 * mse(est, sb)
                loss.backward()
                opt.step()
        m._red.eval()
        return m
