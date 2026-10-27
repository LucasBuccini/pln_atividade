"""Seção 3 do artigo: estatísticas descritivas do corpus e evidência do viés de tamanho.

    python -m fakebr.exploracao

Gera em resultados/tabelas:
  - corpus_tamanho.md       palavras por notícia, por classe e versão do corpus
  - corpus_categorias.md    notícias por categoria em cada partição
  - corpus_fontes.md        quantas fontes distintas há em cada classe e as mais frequentes
e em resultados/figuras:
  - distribuicao_tamanho    histograma de palavras por classe, nas duas versões
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, dados, estilo


def tabela_markdown(df: pd.DataFrame, caminho, indice: bool = True) -> None:
    caminho.write_text(df.to_markdown(index=indice) + "\n", encoding="utf-8")


def quadro(versao: str) -> pd.DataFrame:
    partes = dados.carregar_particoes(versao)
    linhas = []
    for parte, corpus in partes.items():
        for d in corpus:
            linhas.append({"versao": versao, "parte": parte, "rotulo": d.rotulo, **d.metadados})
    return pd.DataFrame(linhas)


def main() -> None:
    config.garantir_pastas()
    df = pd.concat([quadro("completo"), quadro("normalizado")], ignore_index=True)
    df["classe"] = df["rotulo"].map(estilo.NOME_CLASSE)

    # Tamanho por classe e versão: o viés do corpus em números.
    tam = (df.groupby(["versao", "classe"])["palavras_original"]
             .agg(n="count", media="mean", mediana="median", desvio="std", minimo="min", maximo="max")
             .round(1))
    tabela_markdown(tam, config.TABELAS / "corpus_tamanho.md")

    # Diferença de tamanho dentro de cada par (verdadeira - falsa).
    for versao in ("completo", "normalizado"):
        p = df[df.versao == versao].pivot_table(index="par", columns="rotulo", values="palavras_original")
        dif = (p["true"] - p["fake"]).dropna()
        print(f"{versao}: verdadeira mais longa em {np.mean(dif > 0):.1%} dos pares; "
              f"empate em {np.mean(dif == 0):.1%}; razão mediana {np.median(p['true'] / p['fake']):.2f}")

    norm = df[df.versao == "normalizado"]
    cat = pd.crosstab(norm["categoria"], norm["parte"], margins=True, margins_name="total")
    cat = cat[[*config.PARTES, "total"]].sort_values("total", ascending=False)
    tabela_markdown(cat, config.TABELAS / "corpus_categorias.md")

    fontes = norm.groupby("classe")["fonte"].agg(fontes_distintas="nunique").join(
        norm.groupby("classe")["fonte"].agg(
            mais_frequentes=lambda s: ", ".join(f"{f} ({n})" for f, n in s.value_counts().head(5).items())))
    tabela_markdown(fontes, config.TABELAS / "corpus_fontes.md")

    figura_tamanho(df)
    print(tam, "\n", cat, "\n", fontes, sep="")


def figura_tamanho(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    estilo.aplicar()
    fig, eixos = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    bordas = np.logspace(np.log10(5), np.log10(8000), 45)
    for eixo, versao, titulo in zip(eixos, ("completo", "normalizado"),
                                     ("Textos completos", "Textos normalizados")):
        sub = df[df.versao == versao]
        for rotulo in ("true", "fake"):
            valores = sub.loc[sub.rotulo == rotulo, "palavras_original"]
            eixo.hist(valores, bins=bordas, histtype="step", linewidth=2, color=estilo.COR_CLASSE[rotulo],
                      label=f"{estilo.NOME_CLASSE[rotulo]} (mediana {valores.median():.0f})")
        eixo.set_xscale("log")
        eixo.set_title(titulo, loc="left")
        eixo.set_xlabel("Palavras por notícia (escala log)")
        # Traços em vez de caixas; legenda abaixo do eixo para não cobrir as curvas.
        linhas = [Line2D([], [], color=h.get_edgecolor(), linewidth=2) for h in eixo.patches[-2:]]
        eixo.legend(linhas, [t.get_label() for t in eixo.patches[-2:]], loc="upper center",
                    bbox_to_anchor=(0.5, -0.3), ncol=1)
    eixos[0].set_ylabel("Notícias")
    estilo.salvar(fig, "distribuicao_tamanho", config.FIGURAS)
    plt.close(fig)


if __name__ == "__main__":
    main()
