"""Verifica os modelos PUBLICADOS (Keras .h5) do paper, sem TensorFlow.

Reconstrói o forward do RNN em numpy a partir dos pesos (TF não tem wheel p/ Python 3.13)
e roda no X_test/Y_test oficiais. Reproduz a Tabela VII do paper.

Uso: python -m src.verify_published
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np

from . import data as datamod
from .evaluate import report

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "published_models"


def load_weights(h5_path):
    """Lê os pesos de um modelo Keras .h5 (sem TensorFlow), via h5py.

    Extrai os kernels/biases das duas camadas SimpleRNN e da camada densa.

    Args:
        h5_path: caminho do arquivo .h5 do modelo publicado.

    Returns:
        dict com os arrays:
            k1/rk1/b1  — kernel (195×50), recurrent (50×50), bias (50) da 1ª RNN;
            k2/rk2/b2  — kernel (50×10),  recurrent (10×10), bias (10) da 2ª RNN;
            kd/bd      — kernel (10×5) e bias (5) da camada densa.
    """
    f = h5py.File(h5_path, "r")
    rnn = sorted(k for k in f.keys() if k.startswith("simple_rnn"))
    dense = next(k for k in f.keys() if k.startswith("dense"))
    g = lambda grp, name: np.array(f[f"{grp}/{grp}/{name}"])
    r1, r2 = rnn[0], rnn[1]
    return {
        "k1": g(r1, "kernel:0"), "rk1": g(r1, "recurrent_kernel:0"), "b1": g(r1, "bias:0"),
        "k2": g(r2, "kernel:0"), "rk2": g(r2, "recurrent_kernel:0"), "b2": g(r2, "bias:0"),
        "kd": g(dense, "kernel:0"), "bd": g(dense, "bias:0"),
    }


def forward(X, w):
    """Reproduz em numpy o forward do modelo Keras (SimpleRNN ×2 + Dense).

    A recorrência do SimpleRNN é: h_t = tanh(x_t·kernel + h_{t-1}·recurrent + bias),
    com estado inicial zero. A 1ª RNN devolve todos os passos; a 2ª usa só o último.

    Args:
        X: janelas (N, 60, 195) float.
        w: dicionário de pesos retornado por `load_weights`.

    Returns:
        Logits (N, 5) — pontuação por classe (aplicar argmax/softmax fora).
    """
    N, T, _ = X.shape
    # --- 1ª RNN: percorre os 60 passos guardando a saída de cada um ---
    h1 = np.zeros((N, w["k1"].shape[1]), dtype=np.float32)
    H1 = np.empty((N, T, w["k1"].shape[1]), dtype=np.float32)
    for t in range(T):
        h1 = np.tanh(X[:, t, :] @ w["k1"] + h1 @ w["rk1"] + w["b1"])
        H1[:, t, :] = h1
    # --- 2ª RNN: percorre a sequência da 1ª; fica com o último estado ---
    h2 = np.zeros((N, w["k2"].shape[1]), dtype=np.float32)
    for t in range(T):
        h2 = np.tanh(H1[:, t, :] @ w["k2"] + h2 @ w["rk2"] + w["b2"])
    # --- camada densa final -> logits ---
    return h2 @ w["kd"] + w["bd"]


def main():
    """Roda os 3 modelos publicados no teste oficial e imprime as métricas de cada um."""
    _, _, X_test, y_test = datamod.load()
    print(f"Teste oficial: X={X_test.shape}\n")
    for mp in sorted(glob.glob(str(MODELS_DIR / "*.h5"))):
        w = load_weights(mp)
        y_pred = forward(X_test, w).argmax(1)
        report(y_test, y_pred, title=Path(mp).name)
        print()


if __name__ == "__main__":
    main()
