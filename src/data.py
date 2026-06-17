"""Carga do dataset oficial do paper Alkhatib (SOMEIP_IDS).

Lê as sequências pré-processadas de `data/Sequence_data.zip`:
  X_train (2807, 60, 195) / X_test (2771, 60, 195)  — 60 pacotes × 195 features one-hot
  Y_train / Y_test                                   — rótulos (5 classes)

Obs.: no zip, Y_train vem como inteiros (N,1) e Y_test como one-hot (N,5); normalizamos
ambos para índices de classe inteiros.
"""
from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "Sequence_data.zip"

# Ordem das classes (Tabela I/III do paper)
CLASS_NAMES = ["Normal", "ErrorOnEvent", "ErrorOnError", "MissingResponse", "MissingRequest"]
# Pesos por classe — Adaptive Weighting, Tabela III do paper
CLASS_WEIGHTS = [0.16, 6.68, 10.35, 4.33, 4.86]


def _to_labels(Y) -> np.ndarray:
    Y = np.array(Y)
    if Y.ndim > 1 and Y.shape[1] > 1:   # one-hot
        return Y.argmax(1).astype(np.int64)
    return Y.ravel().astype(np.int64)


def load(zip_path: Path = ZIP):
    """Retorna (X_train, y_train, X_test, y_test) como numpy arrays."""
    with zipfile.ZipFile(zip_path) as z:
        X_train = np.array(pickle.load(z.open("X_train.pickle"))).astype(np.float32)
        X_test = np.array(pickle.load(z.open("X_test.pickle"))).astype(np.float32)
        y_train = _to_labels(pickle.load(z.open("Y_train.pickle")))
        y_test = _to_labels(pickle.load(z.open("Y_test.pickle")))
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load()
    print("X_train:", Xtr.shape, "| X_test:", Xte.shape)
    for split, y in [("train", ytr), ("test", yte)]:
        u, c = np.unique(y, return_counts=True)
        print(split, "classes:", {CLASS_NAMES[i]: int(n) for i, n in zip(u, c)})
