"""Etapas tokens -> tokens.

Recursos pesados (listas do NLTK, modelos do spaCy) só são carregados na primeira vez
que a etapa é aplicada, então criar um Pipeline é sempre instantâneo.
"""
from __future__ import annotations

from collections.abc import Iterable

from .base import EtapaTokens, Tokens
from .etapas_texto import RemoverAcentos

# Nome do idioma na lib -> (nome no NLTK, modelo do spaCy).
# Outros idiomas do NLTK funcionam passando o nome dele direto, ex. idioma="spanish".
IDIOMAS = {
    "portugues": ("portuguese", "pt_core_news_sm"),
    "ingles": ("english", "en_core_web_sm"),
}


def nome_nltk(idioma: str) -> str:
    return IDIOMAS.get(idioma, (idioma, None))[0]


def modelo_spacy(idioma: str) -> str:
    modelo = IDIOMAS.get(idioma, (None, None))[1]
    if modelo is None:
        raise ValueError(f"Sem modelo spaCy padrão para {idioma!r}. Use Lematizar(modelo=...) diretamente.")
    return modelo


_sem_acentos = RemoverAcentos().aplicar


def _forma_comparavel(palavra: str) -> str:
    """Minúsculas e sem acentos, para "NÃO", "não" e "nao" serem a mesma palavra."""
    return _sem_acentos(palavra.lower())


class RemoverStopwords(EtapaTokens):
    """Remove palavras muito frequentes e pouco informativas (artigos, preposições...).

    Usa a lista do NLTK para `idioma`. A comparação ignora maiúsculas e acentos, então
    funciona antes ou depois de ParaMinusculas/RemoverAcentos.

    `adicionar`: palavras a remover além da lista padrão, ex. ["governo"].
    `manter`:    palavras da lista que devem ficar, ex. ["não", "nem", "sem"] em análise
                 de sentimentos, onde a negação inverte o sentido da frase.
    """

    def __init__(self, idioma: str = "portugues", adicionar: Iterable[str] = (), manter: Iterable[str] = ()):
        self.idioma = idioma
        self.adicionar = adicionar
        self.manter = manter
        self._lista: set[str] | None = None

    @property
    def lista(self) -> set[str]:
        """Conjunto final de stopwords (já sem acentos e em minúsculas)."""
        if self._lista is None:
            from nltk.corpus import stopwords
            palavras = set(stopwords.words(nome_nltk(self.idioma))) | set(self.adicionar)
            self._lista = {_forma_comparavel(p) for p in palavras} - {_forma_comparavel(p) for p in self.manter}
        return self._lista

    def aplicar(self, tokens: Tokens) -> Tokens:
        return [t for t in tokens if _forma_comparavel(t) not in self.lista]


class RemoverTokensCurtos(EtapaTokens):
    """Descarta tokens com menos de `tamanho_minimo` caracteres (sobras como "a", "x", "q")."""

    def __init__(self, tamanho_minimo: int = 2):
        self.tamanho_minimo = tamanho_minimo

    def aplicar(self, tokens: Tokens) -> Tokens:
        return [t for t in tokens if len(t) >= self.tamanho_minimo]


class ReduzirAoRadical(EtapaTokens):
    """Stemming: corta sufixos para juntar variações da palavra: "governo", "governar" -> "govern".

    É rápido, mas o radical nem sempre é uma palavra real.
    `algoritmo`: "rslp" (específico para português) ou "snowball". Se não informado,
    usa RSLP para português e Snowball para os outros idiomas.
    """

    def __init__(self, idioma: str = "portugues", algoritmo: str | None = None):
        self.idioma = idioma
        self.algoritmo = algoritmo
        self._stemmer = None

    def _criar_stemmer(self):
        from nltk.stem import RSLPStemmer, SnowballStemmer
        algoritmo = self.algoritmo or ("rslp" if nome_nltk(self.idioma) == "portuguese" else "snowball")
        if algoritmo == "rslp":
            return RSLPStemmer()
        if algoritmo == "snowball":
            return SnowballStemmer(nome_nltk(self.idioma))
        raise ValueError(f"Algoritmo de stemming desconhecido: {algoritmo!r}. Use 'rslp' ou 'snowball'.")

    def aplicar(self, tokens: Tokens) -> Tokens:
        if self._stemmer is None:
            self._stemmer = self._criar_stemmer()
        return [self._stemmer.stem(t) for t in tokens]


class Lematizar(EtapaTokens):
    """Troca cada token pela forma de dicionário: "meninos" -> "menino", "estavam" -> "estar".

    Usa spaCy. Mantém a tokenização recebida (um lema por token), mas o modelo vê a frase
    inteira, então decide a classe gramatical pelo contexto. Mais lento e mais preciso que
    ReduzirAoRadical. Use antes de RemoverAcentos e de RemoverStopwords, porque o modelo
    foi treinado com texto acentuado e completo.

    `modelo`: nome do modelo spaCy instalado, ex. "pt_core_news_sm" ou "pt_core_news_lg".
    `tamanho_lote`: quantos documentos o spaCy processa por vez.
    """

    def __init__(self, modelo: str = "pt_core_news_sm", tamanho_lote: int = 256):
        self.modelo = modelo
        self.tamanho_lote = tamanho_lote
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.modelo, exclude=["parser", "ner"])
        return self._nlp

    def aplicar(self, tokens: Tokens) -> Tokens:
        return self.aplicar_em_lote([tokens])[0]

    def aplicar_em_lote(self, documentos: list[Tokens]) -> list[Tokens]:
        from spacy.tokens import Doc
        docs = (Doc(self.nlp.vocab, words=tokens) for tokens in documentos)
        return [[t.lemma_ or t.text for t in doc] for doc in self.nlp.pipe(docs, batch_size=self.tamanho_lote)]
