"""Obtenção do FAKE.BR-Corpus, associação dos metadados e particionamento fixo por pares.

Uso pela linha de comando (baixa o corpus se preciso e cria `dados/particoes.json`):

    python -m fakebr.dados

Decisões que entram na seção "Corpus" do artigo:

- Versões: `normalizado` (size_normalized_texts, padrão dos experimentos: em cada par, o
  texto mais longo é truncado ao número de palavras do mais curto) e `completo` (full_texts,
  usado só para medir o viés de tamanho).
- Pares: a notícia falsa N e a verdadeira N tratam do mesmo assunto. O particionamento é
  feito sobre os pares, então as duas sempre caem na mesma parte e o tema de uma notícia de
  teste nunca é visto no treino pela sua contraparte.
- Numeração: as duas versões não têm exatamente os mesmos números de par (full_texts não
  tem 697 e 1468; size_normalized_texts não tem 586 e 1607; ambas vão até 3602). O número N
  é a mesma notícia nas duas versões, então os metadados de full_texts valem para a versão
  normalizada, exceto nos pares 697 e 1468, que ficam com categoria "desconhecida".
"""
from __future__ import annotations

import json
import random
import subprocess
from dataclasses import replace
from urllib.parse import urlparse

from pln_prep import CarregadorPastas, Corpus

from . import config

VERSOES = {"normalizado": "size_normalized_texts", "completo": "full_texts"}

# Linhas do arquivo N-meta.txt que interessam (formato descrito no README do corpus).
_LINHAS_META = {"autor": 0, "link": 1, "categoria": 2, "data": 3}


def baixar() -> None:
    """Clona o repositório oficial do corpus em `dados/Fake.br-Corpus`, se ainda não existir."""
    if config.CORPUS.is_dir():
        return
    config.DADOS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", config.URL_CORPUS, str(config.CORPUS)], check=True)


def ler_metadados(par: int, rotulo: str) -> dict:
    """Autor, categoria, data e fonte (domínio do link) de uma notícia; "desconhecida" se faltar."""
    caminho = config.CORPUS / "full_texts" / f"{rotulo}-meta-information" / f"{par}-meta.txt"
    if not caminho.exists():
        return {"categoria": "desconhecida", "fonte": "desconhecida", "data": "", "autor": ""}
    linhas = caminho.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    meta = {campo: linhas[i].strip() for campo, i in _LINHAS_META.items()}
    meta["fonte"] = urlparse(meta.pop("link")).netloc.removeprefix("www.") or "desconhecida"
    return meta


def carregar(versao: str = "normalizado") -> Corpus:
    """Carrega uma versão do corpus com os metadados de cada notícia.

    Cada documento tem id "fake/12.txt", rótulo "fake" ou "true" e metadados
    {"par": 12, "categoria", "fonte", "data", "autor", "palavras_original"}.
    `palavras_original` é o tamanho na versão lida, guardado antes de qualquer limpeza.
    """
    if versao not in VERSOES:
        raise ValueError(f"versao deve ser uma de {list(VERSOES)}, não {versao!r}")
    baixar()
    # utf-8-sig remove o BOM que aparece no início de parte dos arquivos de full_texts.
    corpus = CarregadorPastas(config.CORPUS / VERSOES[versao], rotulos=["fake", "true"],
                              codificacao="utf-8-sig", nome=f"FAKE.BR ({versao})").carregar()
    documentos = []
    for d in corpus:
        par = int(d.metadados["arquivo"])
        meta = {"par": par, **ler_metadados(par, d.rotulo), "palavras_original": len(d.texto.split())}
        documentos.append(replace(d, texto=d.texto.strip(), metadados=meta))
    return Corpus(documentos, corpus.nome)


def criar_particoes(sobrescrever: bool = False) -> dict:
    """Sorteia, uma única vez, a parte (treino/validação/teste) de cada par e grava em JSON.

    O sorteio é feito sobre os pares da versão normalizada, com cortes exatos de 70/15/15.
    Os pares que só existem na versão completa vão para o treino, para que nenhum deles
    entre em validação ou teste sem ter sido sorteado.
    """
    if config.PARTICOES.exists() and not sobrescrever:
        return json.loads(config.PARTICOES.read_text(encoding="utf-8"))
    baixar()
    pares = {versao: {int(p.stem) for p in (config.CORPUS / pasta / "fake").glob("*.txt")}
             for versao, pasta in VERSOES.items()}
    sorteados = sorted(pares["normalizado"])
    random.Random(config.SEMENTE).shuffle(sorteados)
    n = len(sorteados)
    corte1 = round(n * config.PROPORCOES[0])
    corte2 = corte1 + round(n * config.PROPORCOES[1])
    so_completo = sorted(pares["completo"] - pares["normalizado"])
    particoes = {
        "semente": config.SEMENTE,
        "proporcoes": config.PROPORCOES,
        "observacao": f"pares {so_completo} existem só em full_texts e foram postos no treino",
        "pares": {
            "treino": sorted(sorteados[:corte1] + so_completo),
            "validacao": sorted(sorteados[corte1:corte2]),
            "teste": sorted(sorteados[corte2:]),
        },
    }
    config.PARTICOES.parent.mkdir(parents=True, exist_ok=True)
    config.PARTICOES.write_text(json.dumps(particoes, indent=1), encoding="utf-8")
    return particoes


def dividir(corpus: Corpus) -> dict[str, Corpus]:
    """Separa um corpus carregado nas partes gravadas em `particoes.json`."""
    pares = criar_particoes()["pares"]
    parte_do_par = {par: parte for parte, lista in pares.items() for par in lista}
    return {parte: corpus.filtrar(lambda d, p=parte: parte_do_par.get(d.metadados["par"]) == p)
            for parte in config.PARTES}


def carregar_particoes(versao: str = "normalizado") -> dict[str, Corpus]:
    """Atalho: carrega a versão pedida e já devolve {"treino": ..., "validacao": ..., "teste": ...}."""
    return dividir(carregar(versao))


if __name__ == "__main__":
    baixar()
    criar_particoes()
    for versao in VERSOES:
        partes = carregar_particoes(versao)
        print(carregar(versao).resumo())
        print("  partes:", {nome: len(c) for nome, c in partes.items()}, "\n")
