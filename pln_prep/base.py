"""Contratos das etapas de pré-processamento.

Toda etapa herda de `Etapa` e pertence a um de três tipos, conforme o que recebe e devolve:

- `EtapaTexto`:   texto  -> texto   (ex.: ParaMinusculas, RemoverURLs)
- `Tokenizador`:  texto  -> tokens  (ex.: TokenizarPorRegex)
- `EtapaTokens`:  tokens -> tokens  (ex.: RemoverStopwords, Lematizar)

"Tokens" é sempre uma lista de strings: ["o", "governo", "anunciou"].

Convenção: todo argumento do `__init__` é guardado em um atributo de mesmo nome.
É isso que faz `parametros()`, `repr` e a serialização em config funcionarem
automaticamente em qualquer subclasse, inclusive nas que você criar.
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

Tokens = list[str]

TEXTO = "texto"
TOKENS = "tokens"

# Nome da classe -> classe. Preenchido automaticamente por `Etapa.__init_subclass__`.
_REGISTRO: dict[str, type[Etapa]] = {}


class Etapa(ABC):
    """Uma transformação aplicada a um documento. Classe abstrata: use as subclasses."""

    entrada: str  # TEXTO ou TOKENS
    saida: str    # TEXTO ou TOKENS

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _REGISTRO[cls.__name__] = cls

    @abstractmethod
    def aplicar(self, documento):
        """Processa um único documento (texto ou lista de tokens, conforme `entrada`)."""

    def aplicar_em_lote(self, documentos: list) -> list:
        """Processa vários documentos. Sobrescreva quando processar em lote for mais rápido (ex.: spaCy)."""
        return [self.aplicar(doc) for doc in documentos]

    def parametros(self) -> dict[str, Any]:
        """Argumentos com que a etapa foi criada, ex.: {"manter": "!?"}."""
        assinatura = inspect.signature(type(self).__init__).parameters.values()
        # Etapas sem __init__ próprio herdam o de `object` (*args, **kwargs): não têm parâmetros.
        nomes = (p.name for p in assinatura
                 if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
        return {nome: getattr(self, nome) for nome in nomes}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Mesmo que `parametros()`. Existe porque o scikit-learn exige esse nome."""
        return self.parametros()

    def para_config(self) -> dict[str, Any]:
        """Dicionário serializável em JSON que permite recriar a etapa com `etapa_de_config`."""
        params = {k: list(v) if isinstance(v, (tuple, set, frozenset)) else v
                  for k, v in self.parametros().items()}
        return {"etapa": type(self).__name__, "parametros": params}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.parametros().items())
        return f"{type(self).__name__}({params})"


class EtapaTexto(Etapa):
    """texto -> texto. Se vier depois do tokenizador, o Pipeline aplica a cada token."""

    entrada = saida = TEXTO

    @abstractmethod
    def aplicar(self, texto: str) -> str: ...


class Tokenizador(Etapa):
    """texto -> tokens. Marca no Pipeline a passagem de texto corrido para lista de tokens."""

    entrada, saida = TEXTO, TOKENS

    @abstractmethod
    def aplicar(self, texto: str) -> Tokens: ...


class EtapaTokens(Etapa):
    """tokens -> tokens."""

    entrada = saida = TOKENS

    @abstractmethod
    def aplicar(self, tokens: Tokens) -> Tokens: ...


def etapa_de_config(config: dict[str, Any]) -> Etapa:
    """Recria uma etapa a partir do dicionário gerado por `Etapa.para_config`."""
    nome = config["etapa"]
    if nome not in _REGISTRO:
        raise KeyError(f"Etapa desconhecida: {nome!r}. Importe o módulo que a define antes de carregar a config.")
    return _REGISTRO[nome](**config.get("parametros", {}))
