"""Pipeline: encadeia etapas, valida a ordem e funciona dentro do scikit-learn."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .base import TEXTO, TOKENS, Etapa, EtapaTexto, etapa_de_config
from .documento import Corpus, como_texto


class Pipeline:
    """Sequência de etapas aplicada, em ordem, a cada documento.

    A ordem é conferida na criação: etapas de tokens exigem um Tokenizador antes delas,
    e o erro aparece na hora, não no meio do processamento. Etapas de texto podem vir
    depois do tokenizador; nesse caso são aplicadas a cada token e os tokens que ficarem
    vazios são descartados.

    `etapas`: lista de objetos Etapa.
    `saida`:  "texto" devolve os tokens unidos por espaço (pronto para TfidfVectorizer);
              "tokens" devolve a lista de tokens de cada documento.

    Métodos em inglês (fit, transform, fit_transform, get_params, set_params) existem só
    para o Pipeline poder ser a primeira etapa de um `sklearn.pipeline.Pipeline`.
    """

    def __init__(self, etapas: list[Etapa], saida: str = TEXTO):
        self.etapas = etapas  # sem copiar: o clone do scikit-learn compara por identidade
        self.saida = saida
        self.validar()

    def validar(self) -> None:
        """Confere tipos e ordem das etapas. Chamado automaticamente na criação."""
        if self.saida not in (TEXTO, TOKENS):
            raise ValueError(f"saida deve ser {TEXTO!r} ou {TOKENS!r}, não {self.saida!r}")
        atual = TEXTO
        for i, etapa in enumerate(self.etapas):
            if not isinstance(etapa, Etapa):
                raise TypeError(f"A etapa {i} não é uma Etapa: {etapa!r}")
            if isinstance(etapa, EtapaTexto):
                continue
            if etapa.entrada != atual:
                dica = "adicione um Tokenizador antes dela" if atual == TEXTO else "o texto já foi tokenizado"
                raise ValueError(f"A etapa {i} ({etapa!r}) espera {etapa.entrada!r} mas recebe {atual!r}: {dica}.")
            atual = etapa.saida
        if self.saida == TOKENS and atual == TEXTO:
            raise ValueError("saida='tokens' exige um Tokenizador no pipeline.")

    def transformar(self, documentos: Iterable | Corpus) -> list:
        """Processa vários documentos e devolve uma lista de resultados, na mesma ordem.

        Aceita lista de strings, pandas.Series ou Corpus. Valores ausentes viram "".
        Cada etapa processa o lote inteiro de uma vez, o que acelera etapas como Lematizar.
        """
        textos = documentos.textos if isinstance(documentos, Corpus) else documentos
        docs: list = [como_texto(t) for t in textos]
        atual = TEXTO
        for etapa in self.etapas:
            if isinstance(etapa, EtapaTexto) and atual == TOKENS:
                docs = [[t for t in etapa.aplicar_em_lote(tokens) if t.strip()] for tokens in docs]
            else:
                docs = etapa.aplicar_em_lote(docs)
                atual = etapa.saida
        if atual == TOKENS and self.saida == TEXTO:
            docs = [" ".join(tokens) for tokens in docs]
        return docs

    def __call__(self, texto: str) -> str | list[str]:
        """Processa um único texto: pipeline("Olá, MUNDO!")."""
        return self.transformar([texto])[0]

    def descrever(self) -> str:
        """Lista numerada das etapas e seus parâmetros, útil para registrar no relatório."""
        return repr(self)

    # Persistência: salve junto com os resultados de cada experimento
    def para_config(self) -> dict[str, Any]:
        return {"saida": self.saida, "etapas": [e.para_config() for e in self.etapas]}

    @classmethod
    def de_config(cls, config: dict[str, Any]) -> Pipeline:
        return cls([etapa_de_config(e) for e in config["etapas"]], saida=config.get("saida", TEXTO))

    def salvar(self, caminho: str | Path) -> None:
        """Grava a configuração em JSON. Recupere com Pipeline.carregar(caminho)."""
        Path(caminho).write_text(json.dumps(self.para_config(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def carregar(cls, caminho: str | Path) -> Pipeline:
        return cls.de_config(json.loads(Path(caminho).read_text(encoding="utf-8")))

    # Compatibilidade com scikit-learn
    def fit(self, X: Iterable, y: Any = None) -> Pipeline:
        return self

    def transform(self, X: Iterable) -> list:
        return self.transformar(X)

    def fit_transform(self, X: Iterable, y: Any = None) -> list:
        return self.transformar(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"etapas": self.etapas, "saida": self.saida}

    def set_params(self, **params: Any) -> Pipeline:
        for nome, valor in params.items():
            setattr(self, nome, valor)
        self.validar()
        return self

    def __repr__(self) -> str:
        etapas = "".join(f"\n  {i}. {e!r}" for i, e in enumerate(self.etapas))
        return f"Pipeline(saida={self.saida!r}){etapas}"
