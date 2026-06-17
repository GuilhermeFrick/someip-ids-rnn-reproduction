"""Métricas de avaliação compartilhadas (por classe + matriz de confusão)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix,
)

from .data import CLASS_NAMES


def report(y_true: np.ndarray, y_pred: np.ndarray, title: str = "") -> dict:
    """Imprime e retorna as métricas de classificação multiclasse.

    Calcula acurácia, F1 macro e ponderado, precisão/recall/F1 por classe e a matriz de
    confusão; imprime tudo de forma legível e devolve num dicionário.

    Args:
        y_true: rótulos verdadeiros (N,), inteiros 0..4.
        y_pred: rótulos previstos (N,), inteiros 0..4.
        title: título opcional exibido no cabeçalho (ex.: nome do modelo).

    Returns:
        dict com chaves: accuracy, f1_macro, f1_weighted, precision, recall, f1
        (listas por classe) e confusion_matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    if title:
        print(f"=== {title} ===")
    print(f"Accuracy: {acc*100:.2f}%  | F1 macro: {macro:.3f}  | F1 weighted: {weighted:.3f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:16s} P={p[i]:.2f} R={r[i]:.2f} F1={f1[i]:.2f}")
    print("  Matriz de confusão (linha=verdadeiro; N/EoEv/EoEr/MResp/MReq):")
    for row in cm:
        print("   ", row.tolist())
    return {"accuracy": acc, "f1_macro": macro, "f1_weighted": weighted,
            "precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(),
            "confusion_matrix": cm.tolist()}
