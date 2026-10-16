"""Tokenizadores: texto -> lista de tokens."""
from __future__ import annotations

import re

from .base import Tokenizador, Tokens


class TokenizarPorEspacos(Tokenizador):
    """Separa o texto nos espaços: "olá, mundo!" -> ["olá,", "mundo!"]. A pontuação fica grudada."""

    def aplicar(self, texto: str) -> Tokens:
        return texto.split()


class TokenizarPorRegex(Tokenizador):
    """Extrai como token cada trecho que casar com `padrao`.

    O padrão padrão pega palavras e números e mantém palavras com hífen:
    "o guarda-chuva, e o sol" -> ["o", "guarda-chuva", "e", "o", "sol"].
    """

    def __init__(self, padrao: str = r"\w+(?:-\w+)*"):
        self.padrao = padrao
        self._regex = re.compile(padrao)

    def aplicar(self, texto: str) -> Tokens:
        return self._regex.findall(texto)
