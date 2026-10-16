"""Etapas texto -> texto.

Também funcionam depois do tokenizador: nesse caso o Pipeline aplica a etapa a cada
token e descarta os tokens que ficarem vazios.
"""
from __future__ import annotations

import html
import re
import unicodedata

from .base import EtapaTexto


class ParaMinusculas(EtapaTexto):
    """Converte para minúsculas: "Governo AÇÃO" -> "governo ação"."""

    def aplicar(self, texto: str) -> str:
        return texto.lower()


class SubstituirPadrao(EtapaTexto):
    """Substitui tudo que casar com a expressão regular `padrao` por `substituir_por`.

    É a base das remoções de URL, e-mail e número, e serve para criar remoções novas
    sem escrever uma classe: SubstituirPadrao(r"@\\w+", " ") remove menções do Twitter.
    """

    def __init__(self, padrao: str, substituir_por: str = " "):
        self.padrao = padrao
        self.substituir_por = substituir_por
        self._regex = re.compile(padrao)

    def aplicar(self, texto: str) -> str:
        return self._regex.sub(self.substituir_por, texto)


class RemoverURLs(SubstituirPadrao):
    """Remove endereços web (http://, https://, www.). Use substituir_por=" URL " para marcar o lugar."""

    def __init__(self, substituir_por: str = " "):
        # Para antes da pontuação final ("veja https://a.com." mantém o ponto da frase).
        super().__init__(r"(?:https?://|www\.)\S+?(?=[.,;:!?)\]\"']*(?:\s|$))", substituir_por)


class RemoverEmails(SubstituirPadrao):
    """Remove endereços de e-mail: "fale com ana@usp.br" -> "fale com  "."""

    def __init__(self, substituir_por: str = " "):
        super().__init__(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", substituir_por)


class RemoverNumeros(SubstituirPadrao):
    """Remove números, inclusive com separadores: "R$ 3.500,00 em 2024" -> "R$   em  "."""

    def __init__(self, substituir_por: str = " "):
        super().__init__(r"\d+(?:[.,]\d+)*", substituir_por)


class RemoverHTML(EtapaTexto):
    """Remove tags HTML e converte entidades: "<b>a</b>&amp;b" -> " a &b"."""

    _tags = re.compile(r"<[^>]+>")

    def aplicar(self, texto: str) -> str:
        return html.unescape(self._tags.sub(" ", texto))


class RemoverPontuacao(EtapaTexto):
    """Troca pontuação e símbolos por espaço, para não colar palavras: "fim.início" -> "fim início".

    `manter`: caracteres a preservar, ex. "!?" em análise de sentimentos ou "-" para
    manter palavras compostas como "guarda-chuva".
    """

    def __init__(self, manter: str = ""):
        self.manter = manter

    def aplicar(self, texto: str) -> str:
        return "".join(c if c.isalnum() or c.isspace() or c in self.manter else " " for c in texto)


class RemoverAcentos(EtapaTexto):
    """Remove acentos e cedilha: "ação" -> "acao". Se for lematizar, use depois do Lematizar."""

    def aplicar(self, texto: str) -> str:
        decomposto = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in decomposto if not unicodedata.combining(c))


class NormalizarEspacos(EtapaTexto):
    """Junta espaços, tabs e quebras de linha repetidos em um único espaço e apara as pontas."""

    def aplicar(self, texto: str) -> str:
        return " ".join(texto.split())
