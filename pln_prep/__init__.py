"""pln_prep: pré-processamento de texto padronizado para qualquer dataset de PLN.

Fluxo completo (documentação detalhada no README.md do repositório):

    from pln_prep import CarregadorCSV, pipelines_prontos

    corpus = CarregadorCSV("dados.csv", coluna_texto="texto", coluna_rotulo="classe").carregar()
    print(corpus.resumo())

    pipeline = pipelines_prontos.para_modelos_classicos("portugues", normalizacao="radical")
    limpo = corpus.preprocessar(pipeline)
    treino, validacao, teste = limpo.dividir((0.7, 0.15, 0.15))

    pipeline.salvar("resultados/pipeline.json")
"""
from . import pipelines_prontos
from .base import TEXTO, TOKENS, Etapa, EtapaTexto, EtapaTokens, Tokenizador, etapa_de_config
from .carregadores import (Carregador, CarregadorCSV, CarregadorDataFrame, CarregadorHuggingFace,
                           CarregadorJSON, CarregadorPastas, CarregadorTabular)
from .documento import Corpus, Documento
from .etapas_texto import (NormalizarEspacos, ParaMinusculas, RemoverAcentos, RemoverEmails, RemoverHTML,
                           RemoverNumeros, RemoverPontuacao, RemoverURLs, SubstituirPadrao)
from .etapas_tokens import IDIOMAS, Lematizar, ReduzirAoRadical, RemoverStopwords, RemoverTokensCurtos
from .pipeline import Pipeline
from .tokenizadores import TokenizarPorEspacos, TokenizarPorRegex

__all__ = [
    # Formato padrão
    "Documento", "Corpus",
    # Carregadores
    "Carregador", "CarregadorTabular", "CarregadorCSV", "CarregadorJSON", "CarregadorDataFrame",
    "CarregadorHuggingFace", "CarregadorPastas",
    # Pipeline
    "Pipeline", "pipelines_prontos",
    # Tipos de etapa
    "Etapa", "EtapaTexto", "Tokenizador", "EtapaTokens", "etapa_de_config", "TEXTO", "TOKENS",
    # Etapas de texto
    "ParaMinusculas", "SubstituirPadrao", "RemoverURLs", "RemoverEmails", "RemoverNumeros",
    "RemoverHTML", "RemoverPontuacao", "RemoverAcentos", "NormalizarEspacos",
    # Tokenizadores
    "TokenizarPorEspacos", "TokenizarPorRegex",
    # Etapas de tokens
    "RemoverStopwords", "RemoverTokensCurtos", "ReduzirAoRadical", "Lematizar", "IDIOMAS",
]
