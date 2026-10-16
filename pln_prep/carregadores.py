"""Carregadores: leem um dataset em qualquer formato e devolvem um `Corpus` no formato padrão.

Todos seguem o mesmo uso:

    corpus = CarregadorCSV("avaliacoes.csv", coluna_texto="review_comment_message",
                           coluna_rotulo="review_score").carregar()

Formatos tabulares (CSV, JSON, DataFrame, HuggingFace) compartilham os mesmos parâmetros
de mapeamento de colunas, definidos em `CarregadorTabular`.
"""
from __future__ import annotations

import csv
import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .documento import Corpus, Documento


class Carregador(ABC):
    """Base de todos os carregadores. Subclasses implementam `carregar()`."""

    @abstractmethod
    def carregar(self) -> Corpus:
        """Lê a origem e devolve um Corpus."""


class CarregadorTabular(Carregador):
    """Base dos formatos em que cada documento é um registro com colunas (dicionário).

    `coluna_texto`:      coluna com o texto. Obrigatória.
    `coluna_rotulo`:     coluna com o rótulo, ou None se a tarefa não tem rótulo.
    `coluna_id`:         coluna com um identificador único, ou None para numerar 0, 1, 2...
    `colunas_metadados`: colunas a guardar em `metadados`. None guarda todas as que sobrarem;
                         [] não guarda nenhuma.
    `nome`:              nome do corpus, usado em `resumo()` e `repr`.
    """

    def __init__(self, coluna_texto: str, coluna_rotulo: str | None = None, coluna_id: str | None = None,
                 colunas_metadados: list[str] | None = None, nome: str = ""):
        self.coluna_texto = coluna_texto
        self.coluna_rotulo = coluna_rotulo
        self.coluna_id = coluna_id
        self.colunas_metadados = colunas_metadados
        self.nome = nome

    @abstractmethod
    def ler_registros(self) -> Iterable[dict[str, Any]]:
        """Devolve os registros da origem, cada um como dicionário coluna -> valor."""

    def converter_rotulo(self, valor: Any) -> Any:
        """Gancho para transformar o rótulo bruto (ex.: índice numérico -> nome da classe)."""
        return valor

    def carregar(self) -> Corpus:
        return Corpus((self.para_documento(i, r) for i, r in enumerate(self.ler_registros())),
                      self.nome or self.nome_padrao())

    def nome_padrao(self) -> str:
        return ""

    def para_documento(self, posicao: int, registro: dict[str, Any]) -> Documento:
        if self.coluna_texto not in registro:
            raise KeyError(f"coluna de texto {self.coluna_texto!r} não encontrada. "
                           f"Colunas disponíveis: {list(registro)}")
        usadas = {self.coluna_texto, self.coluna_rotulo, self.coluna_id}
        if self.colunas_metadados is None:
            metadados = {k: v for k, v in registro.items() if k not in usadas}
        else:
            metadados = {k: registro.get(k) for k in self.colunas_metadados}
        # Arquivos gravados por Corpus.salvar_jsonl têm os metadados aninhados num campo próprio.
        if isinstance(metadados.get("metadados"), dict):
            metadados = {**metadados.pop("metadados"), **metadados}
        return Documento(
            id=registro[self.coluna_id] if self.coluna_id else posicao,
            texto=registro[self.coluna_texto],
            rotulo=self.converter_rotulo(registro[self.coluna_rotulo]) if self.coluna_rotulo else None,
            metadados=metadados,
        )


class CarregadorCSV(CarregadorTabular):
    """Lê um arquivo CSV (ou TSV, com separador="\\t"). Ex.: Brazilian E-Commerce (Olist), Kaggle Resume."""

    def __init__(self, caminho: str | Path, coluna_texto: str, coluna_rotulo: str | None = None,
                 coluna_id: str | None = None, colunas_metadados: list[str] | None = None,
                 separador: str = ",", codificacao: str = "utf-8", nome: str = ""):
        super().__init__(coluna_texto, coluna_rotulo, coluna_id, colunas_metadados, nome)
        self.caminho = caminho
        self.separador = separador
        self.codificacao = codificacao

    def nome_padrao(self) -> str:
        return Path(self.caminho).stem

    def ler_registros(self) -> Iterable[dict[str, Any]]:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # textos longos (notícias, acórdãos)
        with open(self.caminho, encoding=self.codificacao, newline="") as f:
            yield from csv.DictReader(f, delimiter=self.separador)


class CarregadorJSON(CarregadorTabular):
    """Lê JSON (lista de objetos) ou JSON Lines (um objeto por linha, extensão .jsonl).

    Os padrões de coluna correspondem ao formato gravado por `Corpus.salvar_jsonl`, então
    CarregadorJSON("corpus.jsonl").carregar() recupera um corpus salvo sem mais argumentos.
    """

    def __init__(self, caminho: str | Path, coluna_texto: str = "texto", coluna_rotulo: str | None = "rotulo",
                 coluna_id: str | None = "id", colunas_metadados: list[str] | None = None,
                 codificacao: str = "utf-8", nome: str = ""):
        super().__init__(coluna_texto, coluna_rotulo, coluna_id, colunas_metadados, nome)
        self.caminho = caminho
        self.codificacao = codificacao

    def nome_padrao(self) -> str:
        return Path(self.caminho).stem

    def ler_registros(self) -> Iterable[dict[str, Any]]:
        with open(self.caminho, encoding=self.codificacao) as f:
            if str(self.caminho).endswith(".jsonl"):
                yield from (json.loads(linha) for linha in f if linha.strip())
            else:
                yield from json.load(f)


class CarregadorDataFrame(CarregadorTabular):
    """Converte um pandas.DataFrame já carregado (útil quando a leitura exige ajustes do pandas)."""

    def __init__(self, dataframe, coluna_texto: str, coluna_rotulo: str | None = None,
                 coluna_id: str | None = None, colunas_metadados: list[str] | None = None, nome: str = ""):
        super().__init__(coluna_texto, coluna_rotulo, coluna_id, colunas_metadados, nome)
        self.dataframe = dataframe

    def ler_registros(self) -> Iterable[dict[str, Any]]:
        return self.dataframe.to_dict("records")


class CarregadorHuggingFace(CarregadorTabular):
    """Lê um dataset do HuggingFace Hub (ex.: Banking77, XL-Sum) ou um objeto `datasets.Dataset`.

    `dataset`:   nome no Hub (ex. "PolyAI/banking77") ou um Dataset já carregado.
    `divisao`:   "train", "test"... (obrigatório quando `dataset` é um nome).
    `config`:    subconjunto do dataset, ex. "portuguese" no XL-Sum.
    Rótulos numéricos com nomes de classe (ClassLabel) são convertidos para o nome.
    """

    def __init__(self, dataset, coluna_texto: str, coluna_rotulo: str | None = None,
                 coluna_id: str | None = None, colunas_metadados: list[str] | None = None,
                 divisao: str = "train", config: str | None = None, nome: str = ""):
        super().__init__(coluna_texto, coluna_rotulo, coluna_id, colunas_metadados, nome)
        self.dataset = dataset
        self.divisao = divisao
        self.config = config
        self._nomes_classes = None

    def _obter_dataset(self):
        if not isinstance(self.dataset, str):
            return self.dataset
        from datasets import load_dataset
        return load_dataset(self.dataset, self.config, split=self.divisao)

    def nome_padrao(self) -> str:
        return self.dataset if isinstance(self.dataset, str) else ""

    def ler_registros(self) -> Iterable[dict[str, Any]]:
        ds = self._obter_dataset()
        atributo = ds.features.get(self.coluna_rotulo) if self.coluna_rotulo else None
        self._nomes_classes = getattr(atributo, "names", None)
        return iter(ds)

    def converter_rotulo(self, valor: Any) -> Any:
        if self._nomes_classes is not None and isinstance(valor, int):
            return self._nomes_classes[valor]
        return valor


class CarregadorPastas(Carregador):
    """Lê arquivos de texto organizados em uma subpasta por rótulo.

        raiz/
        ├── fake/  1.txt, 2.txt, ...
        └── true/  1.txt, 2.txt, ...

    O nome da subpasta vira o rótulo e o caminho relativo ("fake/1.txt") vira o id.
    Em `metadados` ficam "arquivo" (nome sem extensão, ex. "1") e "pasta".
    Ex.: FAKE.BR-Corpus (`size_normalized_texts` ou `full_texts`).

    `rotulos`: subpastas a ler; None lê todas.
    `padrao`:  quais arquivos ler dentro de cada subpasta.
    """

    def __init__(self, raiz: str | Path, rotulos: list[str] | None = None, padrao: str = "*.txt",
                 codificacao: str = "utf-8", nome: str = ""):
        self.raiz = raiz
        self.rotulos = rotulos
        self.padrao = padrao
        self.codificacao = codificacao
        self.nome = nome

    def carregar(self) -> Corpus:
        raiz = Path(self.raiz)
        if not raiz.is_dir():
            raise FileNotFoundError(f"pasta não encontrada: {raiz}")
        pastas = [raiz / r for r in self.rotulos] if self.rotulos else sorted(p for p in raiz.iterdir() if p.is_dir())
        documentos = []
        for pasta in pastas:
            # Ordena "2.txt" antes de "10.txt" quando os nomes são números.
            arquivos = sorted(pasta.glob(self.padrao), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem))
            for arquivo in arquivos:
                documentos.append(Documento(
                    id=f"{pasta.name}/{arquivo.name}",
                    texto=arquivo.read_text(encoding=self.codificacao),
                    rotulo=pasta.name,
                    metadados={"arquivo": arquivo.stem, "pasta": pasta.name},
                ))
        return Corpus(documentos, self.nome or raiz.name)
