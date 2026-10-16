"""Formato padrão de entrada: todo dataset vira um `Corpus`, que é uma lista de `Documento`.

Qualquer que seja a origem (CSV, JSON, pastas de .txt, DataFrame, HuggingFace), depois
de carregado o dado sempre tem a mesma forma:

    Documento(id="17", texto="O governo anunciou...", rotulo="fake", metadados={"categoria": "politica"})

- `id`:        identificador único, sempre string.
- `texto`:     o texto a ser processado, sempre string ("" se ausente na origem).
- `rotulo`:    a resposta esperada (classe, intenção, sentimento...) ou None em tarefas sem rótulo.
- `metadados`: tudo o mais que a origem tiver (autor, data, resumo de referência...).
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean, median
from typing import Any


def como_texto(valor: Any) -> str:
    """Converte para string; valores ausentes (None, NaN do pandas) viram texto vazio."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ""
    return str(valor)


@dataclass(frozen=True)
class Documento:
    """Um item do dataset. Imutável: o pré-processamento cria documentos novos."""

    id: str
    texto: str
    rotulo: Any = None
    metadados: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "id", como_texto(self.id))
        object.__setattr__(self, "texto", como_texto(self.texto))


class Corpus:
    """Coleção ordenada de Documentos com as operações comuns a qualquer dataset."""

    def __init__(self, documentos: Iterable[Documento], nome: str = ""):
        self.documentos = list(documentos)
        self.nome = nome
        repetidos = [i for i, n in Counter(d.id for d in self.documentos).items() if n > 1]
        if repetidos:
            raise ValueError(f"ids repetidos no corpus: {repetidos[:5]}. Informe uma coluna de id única ou nenhuma.")

    # Acesso
    def __len__(self) -> int:
        return len(self.documentos)

    def __iter__(self) -> Iterator[Documento]:
        return iter(self.documentos)

    def __getitem__(self, i):
        """corpus[0] devolve um Documento; corpus[10:20] devolve um Corpus."""
        if isinstance(i, slice):
            return Corpus(self.documentos[i], self.nome)
        return self.documentos[i]

    @property
    def textos(self) -> list[str]:
        """Lista com o texto de cada documento, na ordem. É o X dos modelos."""
        return [d.texto for d in self.documentos]

    @property
    def rotulos(self) -> list[Any]:
        """Lista com o rótulo de cada documento, na ordem. É o y dos modelos."""
        return [d.rotulo for d in self.documentos]

    @property
    def ids(self) -> list[str]:
        return [d.id for d in self.documentos]

    # Análise
    def contar_rotulos(self) -> dict[Any, int]:
        """Quantos documentos há de cada rótulo, do mais para o menos frequente."""
        return dict(Counter(self.rotulos).most_common())

    def resumo(self) -> str:
        """Texto curto com tamanho do corpus, rótulos e tamanho dos textos em palavras."""
        palavras = [len(t.split()) for t in self.textos] or [0]
        linhas = [
            f"Corpus {self.nome!r}: {len(self)} documentos",
            f"  rótulos: {self.contar_rotulos()}",
            f"  palavras por texto: média {mean(palavras):.0f}, mediana {median(palavras):.0f}, "
            f"mín {min(palavras)}, máx {max(palavras)}",
            f"  textos vazios: {sum(1 for t in self.textos if not t.strip())}",
        ]
        return "\n".join(linhas)

    # Transformações (todas devolvem um Corpus novo; o original não muda)
    def preprocessar(self, pipeline) -> Corpus:
        """Aplica um Pipeline ao texto de todos os documentos. Id, rótulo e metadados são mantidos."""
        if pipeline.saida != "texto":
            raise ValueError("preprocessar exige um Pipeline com saida='texto'. "
                             "Para obter listas de tokens, use pipeline.transformar(corpus).")
        novos = pipeline.transformar(self.textos)
        return Corpus((replace(d, texto=t) for d, t in zip(self.documentos, novos)), self.nome)

    def filtrar(self, condicao: Callable[[Documento], bool]) -> Corpus:
        """Mantém só os documentos para os quais `condicao(documento)` é verdadeira.

        Ex.: corpus.filtrar(lambda d: d.texto.strip() != "") remove textos vazios.
        """
        return Corpus((d for d in self.documentos if condicao(d)), self.nome)

    def dividir(
        self,
        proporcoes: tuple[float, ...] = (0.7, 0.15, 0.15),
        estratificar: bool = True,
        agrupar_por: str | None = None,
        semente: int = 42,
    ) -> tuple[Corpus, ...]:
        """Divide em partes (ex.: treino, validação, teste) de forma reprodutível.

        `proporcoes`:  fração de cada parte; precisa somar 1. Devolve uma parte por valor.
        `estratificar`: mantém a proporção de cada rótulo igual em todas as partes.
        `agrupar_por`: nome de um campo de `metadados`; documentos com o mesmo valor ficam
                       sempre na mesma parte. Ex.: no FAKE.BR, agrupar pelo nome do arquivo
                       mantém cada par falsa/verdadeira junto e evita vazamento de tema.
        `semente`:     a mesma semente gera sempre a mesma divisão.
        """
        if not math.isclose(sum(proporcoes), 1.0, abs_tol=1e-6):
            raise ValueError(f"as proporções precisam somar 1, mas somam {sum(proporcoes)}")

        grupos: dict[Any, list[Documento]] = defaultdict(list)
        for i, d in enumerate(self.documentos):
            chave = d.metadados[agrupar_por] if agrupar_por else i
            grupos[chave].append(d)

        # Estrato de um grupo = conjunto de rótulos dele (um grupo de 1 documento -> o próprio rótulo).
        estratos: dict[Any, list[list[Documento]]] = defaultdict(list)
        for docs in grupos.values():
            chave = tuple(sorted({str(d.rotulo) for d in docs})) if estratificar else None
            estratos[chave].append(docs)

        sorteio = random.Random(semente)
        partes: list[list[Documento]] = [[] for _ in proporcoes]
        for chave in sorted(estratos, key=str):
            lista = estratos[chave]
            sorteio.shuffle(lista)
            limites = [round(len(lista) * sum(proporcoes[:k + 1])) for k in range(len(proporcoes))]
            inicio = 0
            for k, fim in enumerate(limites):
                for docs in lista[inicio:fim]:
                    partes[k].extend(docs)
                inicio = fim
        return tuple(Corpus(p, self.nome) for p in partes)

    # Saída
    def salvar_jsonl(self, caminho: str | Path) -> None:
        """Grava no formato padrão (uma linha JSON por documento). Lido de volta por CarregadorJSON."""
        with open(caminho, "w", encoding="utf-8") as f:
            for d in self.documentos:
                linha = {"id": d.id, "texto": d.texto, "rotulo": d.rotulo, "metadados": d.metadados}
                f.write(json.dumps(linha, ensure_ascii=False, default=str) + "\n")

    def para_dataframe(self):
        """Converte para pandas.DataFrame com colunas id, texto, rotulo e uma por metadado."""
        import pandas as pd
        return pd.DataFrame([{"id": d.id, "texto": d.texto, "rotulo": d.rotulo, **d.metadados}
                             for d in self.documentos])

    def __repr__(self) -> str:
        return f"Corpus(nome={self.nome!r}, documentos={len(self)}, rotulos={self.contar_rotulos()})"
