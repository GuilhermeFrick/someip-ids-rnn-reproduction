"""Treina o RNN do paper Alkhatib DO ZERO (reimplementação PyTorch).

Hiperparâmetros do paper (Tabela V): Adam, lr 1e-3, batch 100, 50 épocas, cross-entropy
categórica, com Adaptive Weighting (pesos por classe da Tabela III) p/ o desbalanceamento.

Treina em X_train e avalia em X_test (split oficial). Uso:
    python -m src.train
    python -m src.train --epochs 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import data as datamod
from .model import AlkhatibRNN, count_parameters
from .evaluate import report

ROOT = Path(__file__).resolve().parents[1]


def run(epochs: int = 50, batch_size: int = 100, lr: float = 1e-3,
        seed: int = 42, save_as: str = "rnn_pytorch.pt", verbose: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, ytr, Xte, yte = datamod.load()
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=batch_size, shuffle=True)

    model = AlkhatibRNN().to(device)
    weights = torch.tensor(datamod.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights)   # Adaptive Weighting (Tabela III)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"Treino: X_train={Xtr.shape} | modelo={count_parameters(model)} params | {device}")

    for ep in range(1, epochs + 1):
        model.train()
        for X, y in tr:
            opt.zero_grad()
            crit(model(X.to(device)), y.to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # estabiliza o SimpleRNN
            opt.step()
        if verbose and (ep % 10 == 0 or ep == 1):
            model.eval()
            with torch.no_grad():
                acc = (model(torch.from_numpy(Xte).to(device)).argmax(1).cpu().numpy() == yte).mean()
            print(f"  época {ep:3d} | test acc: {acc*100:.2f}%")

    # avaliação final
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(Xte).to(device)).argmax(1).cpu().numpy()
    print()
    metrics = report(yte, y_pred, title=f"RNN PyTorch (treino do zero, {epochs} épocas)")

    out = ROOT / "results" / save_as
    out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out)
    if verbose:
        print("\nmodelo salvo em", out)
    return model, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
