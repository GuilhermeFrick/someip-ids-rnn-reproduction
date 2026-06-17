# Reprodução — Alkhatib et al. (2021), SOME/IP IDS com RNN

Repositório **organizado e self-contained** para reproduzir:

> N. Alkhatib, J.-L. Danger, H. Ghauch, *SOME/IP Intrusion Detection using Deep Learning-based
> Sequential Models in Automotive Ethernet Networks*, IEEE IEMCON 2021. (hal-03372353)

O repositório oficial dos autores (`github.com/Alkhatibnatasha/SOMEIP_IDS`) publica os **dados**
e os **modelos treinados**, mas **não inclui código**. Este projeto fornece o código faltante,
em duas frentes:

1. **Verificar os modelos publicados** (`src/verify_published.py`) — roda os 3 RNNs `.h5` do
   paper no teste oficial e reproduz a **Tabela VII** (~98% acc). Feito em **numpy** (sem
   TensorFlow, que não tem wheel p/ Python 3.13).
2. **Treinar do zero** (`src/train.py`) — reimplementação do RNN em **PyTorch**, com os
   hiperparâmetros do paper, treinando em `X_train` e avaliando em `X_test`.

## Estrutura
```
Alkhatib2021-repro/
├── README.md
├── requirements.txt
├── data/
│   ├── Sequence_data.zip          # X_train/X_test (60×195) + Y_train/Y_test (do repo oficial)
│   └── published_models/          # 3 RNNs treinados do paper (.h5 + .json)
├── src/
│   ├── data.py                    # carga do dataset (zip)
│   ├── model.py                   # RNN(195→50→10→5) em PyTorch (= SimpleRNN do paper)
│   ├── train.py                   # treino do zero (Adam, lr 1e-3, batch 100, 50 épocas, pesos por classe)
│   ├── evaluate.py                # métricas por classe + matriz de confusão
│   └── verify_published.py        # inferência numpy nos modelos .h5 publicados
└── results/                       # pesos/saídas geradas
```

## Dataset (do repo oficial)
- `X_train (2807, 60, 195)`, `X_test (2771, 60, 195)` — 60 pacotes × 195 features one-hot.
- 5 classes: `Normal, ErrorOnEvent, ErrorOnError, MissingResponse, MissingRequest`.
- Distribuição bate com a **Tabela I** do paper (treino 2807 / teste 2771).

## Como rodar
```bash
pip install -r requirements.txt

# 1) Verificar os modelos publicados (reproduz a Tabela VII do paper)
python -m src.verify_published

# 2) Treinar o RNN do zero (PyTorch)
python -m src.train --epochs 50
```

## Resultados

### Verificação dos modelos publicados (numpy) — reproduz o paper ✅
| Métrica | rnn…203913 | rnn…204024 | rnn…204132 | Paper (Tab. VII) |
|---------|-----------:|-----------:|-----------:|------------------|
| Accuracy | 98,05% | 97,91% | 98,02% | ~98% |
| F1 Normal | 0,99 | 0,99 | 0,99 | 0,99 |
| F1 ErrorOnError (classe difícil) | 0,73 | 0,81 | 0,86 | ~0,79–0,87 |
| F1 MissingRequest | 0,93 | 0,92 | 0,88 | ~0,93–0,99 |

> Os recalls coincidem com o paper; o padrão é o mesmo (Normal quase perfeito, ErrorOnError é
> a classe mais difícil). **Confirma que o paper é reprodutível ao nível de inferência.**

### Treino do zero (PyTorch) — reprodução próxima ✅
60 épocas, Adam lr 1e-3, batch 100, pesos por classe (Tabela III) + **gradient clipping**
(essencial: sem ele o SimpleRNN explode o gradiente sobre 60 passos e o treino colapsa):

| Classe | F1 (nosso treino) | Modelos publicados |
|--------|------------------:|-------------------:|
| Normal | 0,99 | 0,99 |
| ErrorOnEvent | 0,86 | 0,95–0,98 |
| ErrorOnError | 0,55 | 0,73–0,86 |
| MissingResponse | 0,76 | 0,79–0,85 |
| MissingRequest | 0,87 | 0,88–0,93 |
| **Accuracy** | **97,08%** | ~98% |

Chega perto dos modelos publicados (97% vs 98%). As classes minoritárias (sobretudo
ErrorOnError) ficam um pouco mais fracas — esperado, pois não temos a receita exata dos
autores (inicialização, early stopping, 3-fold CV). A **verificação dos modelos publicados**
(acima) é a reprodução mais fiel; o treino do zero confirma que a arquitetura/método reproduzem.

## Notas
- **Por que não TensorFlow?** Os modelos são Keras `.h5`, mas o TF não tem wheel para
  Python 3.13. Para a verificação, reconstruímos o forward do `SimpleRNN` em numpy a partir
  dos pesos (matematicamente idêntico). Para treinar, usamos PyTorch (`nn.RNN` tanh ≡ SimpleRNN).
- **Atenção a repositórios:** este reproduz o paper de 2021 (5 classes, 195 features). Existe
  um repo posterior (`supervised_detection_some_ip`, binário/58-features) que **não** corresponde
  aos números do paper.
