"""Pacote de reprodução do paper Alkhatib et al. (2021) — SOME/IP IDS com RNN.

Módulos:
    data              Carga do dataset oficial (sequências 60×195, 5 classes).
    model             RNN(195→50→10→5) em PyTorch (equivalente ao SimpleRNN do paper).
    train             Treino do modelo do zero (hiperparâmetros do paper + gradient clipping).
    evaluate          Métricas por classe + matriz de confusão (compartilhado).
    verify_published  Inferência (numpy) nos 3 modelos Keras .h5 publicados pelos autores.

Uso típico:
    python -m src.verify_published     # reproduz a Tabela VII do paper (~98%)
    python -m src.train                # treina do zero (~97%)
"""
