"""Pipelines prontos para as duas famílias de modelos dos trabalhos."""
from __future__ import annotations

from collections.abc import Iterable

from .base import TEXTO
from .etapas_texto import (NormalizarEspacos, ParaMinusculas, RemoverAcentos, RemoverEmails, RemoverHTML,
                           RemoverNumeros, RemoverPontuacao, RemoverURLs)
from .etapas_tokens import Lematizar, ReduzirAoRadical, RemoverStopwords, RemoverTokensCurtos, modelo_spacy
from .pipeline import Pipeline
from .tokenizadores import TokenizarPorRegex


def para_modelos_classicos(
    idioma: str = "portugues",
    normalizacao: str | None = None,
    manter_stopwords: Iterable[str] = (),
    stopwords_extras: Iterable[str] = (),
    remover_acentos: bool = False,
    saida: str = TEXTO,
) -> Pipeline:
    """Limpeza completa para modelos de saco de palavras: TF-IDF/BoW com SVM, Naive Bayes, K-Means...

    Etapas, nesta ordem: RemoverHTML, RemoverURLs, RemoverEmails, ParaMinusculas,
    RemoverNumeros, RemoverPontuacao, NormalizarEspacos, TokenizarPorRegex, RemoverStopwords
    (com Lematizar antes ou ReduzirAoRadical depois, conforme `normalizacao`),
    RemoverAcentos (opcional) e RemoverTokensCurtos.

    `normalizacao`:     None, "radical" (stemming, rápido) ou "lema" (lematização, mais precisa).
    `manter_stopwords`: ex. ["não", "nem", "sem"] para análise de sentimentos.
    `stopwords_extras`: palavras do domínio a remover, ex. ["reais", "produto"].
    `remover_acentos`:  junta "não" e "nao", útil em textos informais escritos sem acento.
    """
    stopwords = RemoverStopwords(idioma, adicionar=stopwords_extras, manter=manter_stopwords)
    etapas = [
        RemoverHTML(), RemoverURLs(), RemoverEmails(), ParaMinusculas(), RemoverNumeros(),
        RemoverPontuacao(), NormalizarEspacos(), TokenizarPorRegex(),
    ]
    # O lematizador precisa das stopwords como contexto para acertar a classe gramatical;
    # o stemmer vem depois delas, porque radicais não casariam com a lista de stopwords.
    if normalizacao == "lema":
        etapas += [Lematizar(modelo_spacy(idioma)), stopwords]
    elif normalizacao == "radical":
        etapas += [stopwords, ReduzirAoRadical(idioma)]
    elif normalizacao is None:
        etapas += [stopwords]
    else:
        raise ValueError(f"normalizacao deve ser None, 'radical' ou 'lema', não {normalizacao!r}")
    if remover_acentos:
        etapas.append(RemoverAcentos())
    etapas.append(RemoverTokensCurtos(2))
    return Pipeline(etapas, saida=saida)


def para_transformers() -> Pipeline:
    """Limpeza mínima para BERT/BERTimbau e LLMs: RemoverHTML, RemoverURLs, RemoverEmails, NormalizarEspacos.

    Esses modelos têm tokenizador próprio e foram pré-treinados com texto natural:
    maiúsculas, acentos, pontuação e stopwords carregam informação e devem ser mantidos.
    """
    return Pipeline([RemoverHTML(), RemoverURLs(), RemoverEmails(), NormalizarEspacos()])
