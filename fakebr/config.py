"""Constantes compartilhadas por todos os experimentos: caminhos, semente e rótulos.

Mudar algo aqui muda todos os experimentos de uma vez; por isso nenhum outro módulo
define caminhos ou sementes por conta própria.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = Path(os.environ.get("FAKEBR_DADOS", RAIZ / "dados"))
CORPUS = DADOS / "Fake.br-Corpus"
PARTICOES = DADOS / "particoes.json"          # versionado no git: a divisão nunca muda
CACHE = DADOS / "cache"                       # textos pré-processados (gerados, fora do git)

RESULTADOS = RAIZ / "resultados"
METRICAS = RESULTADOS / "metricas"            # um JSON por modelo
PREDICOES = RESULTADOS / "predicoes"          # um CSV por modelo (id, rótulo, predição, escore)
TABELAS = RESULTADOS / "tabelas"
FIGURAS = RESULTADOS / "figuras"
MODELOS = RESULTADOS / "modelos"              # pesos treinados (fora do git)

URL_CORPUS = "https://github.com/roneysco/Fake.br-Corpus.git"

SEMENTE = 42
PROPORCOES = (0.70, 0.15, 0.15)               # treino, validação, teste (em pares)
PARTES = ("treino", "validacao", "teste")

POSITIVO = "fake"                             # classe de interesse para precisão/revocação/F1
NEGATIVO = "true"
ROTULOS = (NEGATIVO, POSITIVO)


def fixar_sementes(semente: int = SEMENTE) -> None:
    """Fixa as sementes de random, numpy e torch (se instalado) para resultados repetíveis."""
    random.seed(semente)
    os.environ["PYTHONHASHSEED"] = str(semente)
    try:
        import numpy as np
        np.random.seed(semente)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(semente)
        torch.cuda.manual_seed_all(semente)
    except ImportError:
        pass


def garantir_pastas() -> None:
    for pasta in (CACHE, METRICAS, PREDICOES, TABELAS, FIGURAS, MODELOS):
        pasta.mkdir(parents=True, exist_ok=True)
