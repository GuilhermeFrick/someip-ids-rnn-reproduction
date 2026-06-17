"""Reimplementação em PyTorch do modelo RNN do paper Alkhatib (2021).

Arquitetura (Tabela V do paper / .json dos modelos publicados):
    SimpleRNN(50, tanh, return_sequences) -> SimpleRNN(10, tanh) -> Dense(5, softmax)

Em PyTorch, `nn.RNN(nonlinearity='tanh')` é equivalente ao SimpleRNN do Keras. O softmax
fica implícito na CrossEntropyLoss (o forward retorna logits).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AlkhatibRNN(nn.Module):
    """RNN sequencial do paper Alkhatib (camadas 50 → 10 → 5).

    Equivale ao modelo Keras publicado: duas camadas SimpleRNN (tanh) empilhadas seguidas
    de uma camada densa com softmax. Aqui o softmax fica implícito na CrossEntropyLoss,
    então o `forward` devolve logits.

    Args:
        input_dim: nº de features por pacote (195, após one-hot).
        h1: tamanho oculto da 1ª RNN (50).
        h2: tamanho oculto da 2ª RNN (10).
        n_classes: nº de classes de saída (5).
    """

    def __init__(self, input_dim: int = 195, h1: int = 50, h2: int = 10, n_classes: int = 5):
        super().__init__()
        # 1ª RNN: processa toda a sequência e devolve a saída de TODOS os passos (return_sequences).
        self.rnn1 = nn.RNN(input_dim, h1, batch_first=True, nonlinearity="tanh")
        # 2ª RNN: recebe a sequência da 1ª; usaremos só a saída do último passo.
        self.rnn2 = nn.RNN(h1, h2, batch_first=True, nonlinearity="tanh")
        # Camada densa final: do estado oculto (10) para as 5 classes.
        self.fc = nn.Linear(h2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propaga uma janela de pacotes pela rede.

        Args:
            x: tensor (B, 60, 195) — B janelas de 60 pacotes × 195 features.

        Returns:
            Logits (B, 5) — uma pontuação por classe (sem softmax).
        """
        out1, _ = self.rnn1(x)        # (B, 60, 50) — saída de todos os passos
        out2, _ = self.rnn2(out1)     # (B, 60, 10)
        last = out2[:, -1, :]         # (B, 10) — resumo da sequência (último passo)
        return self.fc(last)          # (B, 5) logits


def count_parameters(model: nn.Module) -> int:
    """Conta os parâmetros treináveis do modelo (para conferir o tamanho da rede)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = AlkhatibRNN()
    x = torch.randn(4, 60, 195)
    print("saída:", tuple(m(x).shape), "| parâmetros:", count_parameters(m))
