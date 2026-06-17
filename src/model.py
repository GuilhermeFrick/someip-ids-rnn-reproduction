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
    def __init__(self, input_dim: int = 195, h1: int = 50, h2: int = 10, n_classes: int = 5):
        super().__init__()
        self.rnn1 = nn.RNN(input_dim, h1, batch_first=True, nonlinearity="tanh")
        self.rnn2 = nn.RNN(h1, h2, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(h2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.rnn1(x)        # (B, 60, 50) — return_sequences
        out2, _ = self.rnn2(out1)     # (B, 60, 10)
        last = out2[:, -1, :]         # (B, 10) — último passo
        return self.fc(last)          # (B, 5) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = AlkhatibRNN()
    x = torch.randn(4, 60, 195)
    print("saída:", tuple(m(x).shape), "| parâmetros:", count_parameters(m))
