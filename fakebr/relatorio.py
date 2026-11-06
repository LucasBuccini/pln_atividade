"""Seções 5 e 6 do artigo: consolida os resultados de todos os experimentos.

    python -m fakebr.relatorio

Lê resultados/metricas/*.json e resultados/predicoes/*.csv e gera, em resultados/tabelas:
  - resultados_teste.md / .tex     tabela principal (teste), com IC95% por bootstrap de pares
  - resultados_validacao.md        a mesma tabela na validação
  - mcnemar.md                     testes pareados entre os modelos principais (correção de Holm)
  - erros_por_categoria.md         acurácia por categoria temática
  - erros_por_fonte.md             acurácia nas notícias falsas do site dominante vs. demais fontes
  - erros_por_tamanho.md           acurácia por quartil de tamanho do texto
  - erros_por_final.md             acurácia por classe e por "termina com pontuação final" (atalho do truncamento)
  - erros_comuns.csv               notícias que todos os modelos principais erram (análise qualitativa)
  - termos_tfidf_svm.md            termos de maior peso da SVM linear
  - robustez_artefatos.md          cada modelo no texto mínimo vs. com os artefatos neutralizados
  - custo.md                       tempo de ajuste e de inferência de cada modelo
e, em resultados/figuras: comparacao_acuracia, robustez_artefatos e matrizes_confusao.

Modelos treinados com várias sementes (campo info.grupo, ex. BERTimbau) aparecem numa linha
agregada com média ± desvio-padrão; nos testes pareados entra a execução da semente 42.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from . import config, estilo
from .avaliacao import mcnemar

# Um representante de cada família, na ordem de complexidade crescente. Os que ainda não
# foram rodados são ignorados.
PRINCIPAIS = ["tfidf_nb", "tfidf_svm", "svm_caracteres", "bertimbau_congelado", "e5_congelado",
              "bertimbau_s42", "qwen7b_k0_calibrado", "qwen7b_k4_calibrado"]
# Os mesmos modelos sobre o texto com os artefatos de formatação neutralizados.
NEUTROS = ["svm_caracteres_neutro", "bertimbau_congelado_neutro", "bertimbau_neutro_s42",
           "qwen7b_neutro_k0_calibrado", "qwen7b_neutro_k4_calibrado"]
# (modelo no texto mínimo, mesmo modelo no texto neutralizado). Nomes de grupo (ex. "bertimbau")
# juntam as sementes; "tfidf_svm" não entra porque já remove toda a pontuação.
PARES_NEUTRO = [("comprimento", "comprimento_neutro"), ("formatacao", "formatacao_neutro"),
                ("svm_caracteres", "svm_caracteres_neutro"), ("bertimbau_congelado", "bertimbau_congelado_neutro"),
                ("bertimbau", "bertimbau_neutro"), ("qwen7b_k0_calibrado", "qwen7b_neutro_k0_calibrado"),
                ("qwen7b_k4_calibrado", "qwen7b_neutro_k4_calibrado")]
SITE_DOMINANTE = "diariodobrasil.org"
# Versão enxuta da figura de acurácia para o artigo: os mesmos modelos da tabela principal.
FIGURA_ARTIGO = ["comprimento_completo", "tfidf_svm_completo", "comprimento", "formatacao", "formatacao_neutro",
                 "tfidf_nb", "tfidf_reglog", "tfidf_svm", "svm_caracteres", "e5_congelado", "bertimbau_congelado",
                 "bertimbau", "qwen7b_k0", "qwen7b_k0_calibrado", "qwen7b_k4", "qwen7b_k4_calibrado"]


# --------------------------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------------------------
def carregar_metricas() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(config.METRICAS.glob("*.json"))}


def carregar_predicoes(modelo: str, parte: str = "teste") -> pd.DataFrame:
    return pd.read_csv(config.PREDICOES / f"{modelo}_{parte}.csv")


def nome_exibicao(modelo: str, registro: dict) -> str:
    return registro["info"].get("descricao", modelo)


# --------------------------------------------------------------------------------------------
# Tabela principal
# --------------------------------------------------------------------------------------------
def _pct(x: float) -> str:
    """Percentual com uma casa e vírgula decimal (o artigo é em português)."""
    return f"{100 * x:.1f}".replace(".", ",")


def _p(x: float) -> str:
    """p-valor com 2 algarismos significativos e vírgula decimal."""
    return f"{x:.2g}".replace(".", ",")


def _num(x, casas: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{casas}f}".replace(".", ",")


def _dec(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def tabela_resultados(registros: dict[str, dict], parte: str) -> pd.DataFrame:
    linhas, grupos = [], {}
    for modelo, reg in registros.items():
        if parte not in reg:
            continue
        info, m = reg["info"], reg[parte]
        if "grupo" in info:
            grupos.setdefault(info["grupo"], []).append(reg)
            continue
        ic = m["ic95"]["acuracia"]
        linhas.append({
            "familia": info.get("familia", ""), "modelo": modelo, "descricao": info.get("descricao", modelo),
            "acuracia": _pct(m["acuracia"]), "ic95": f"[{_pct(ic[0])}; {_pct(ic[1])}]",
            "precisao_fake": _pct(m["precisao_fake"]), "revocacao_fake": _pct(m["revocacao_fake"]),
            "f1_fake": _pct(m["f1_fake"]), "f1_macro": _pct(m["f1_macro"]),
            "roc_auc": _dec(m["roc_auc"]) if "roc_auc" in m else "–", "_ordem": m["acuracia"],
        })
    for grupo, regs in grupos.items():
        def ms(chave, fmt=_pct):
            v = np.array([r[parte][chave] for r in regs])
            return f"{fmt(v.mean())} ± {fmt(v.std(ddof=1)) if len(v) > 1 else '–'}"
        info = regs[0]["info"]
        linhas.append({
            "familia": info["familia"], "modelo": f"{grupo} ({len(regs)} sementes)",
            "descricao": info.get("descricao", grupo), "acuracia": ms("acuracia"),
            "ic95": "média ± dp", "precisao_fake": ms("precisao_fake"), "revocacao_fake": ms("revocacao_fake"),
            "f1_fake": ms("f1_fake"), "f1_macro": ms("f1_macro"),
            "roc_auc": ms("roc_auc", _dec),
            "_ordem": float(np.mean([r[parte]["acuracia"] for r in regs])),
        })
    df = pd.DataFrame(linhas).sort_values(["familia", "_ordem"], ascending=[True, False])
    return df.drop(columns="_ordem").reset_index(drop=True)


COLUNAS = {"descricao": "Modelo", "acuracia": "Acurácia", "ic95": "IC 95%", "precisao_fake": "Precisão",
           "revocacao_fake": "Revocação", "f1_fake": "F1", "f1_macro": "F1 macro", "roc_auc": "ROC-AUC"}


def para_markdown(df: pd.DataFrame) -> str:
    partes = []
    for familia, sub in df.groupby("familia", sort=False):
        partes.append(f"**{familia}**\n\n" + sub[list(COLUNAS)].rename(columns=COLUNAS).to_markdown(index=False))
    return "Precisão, revocação e F1 referem-se à classe *falsa*. Valores em %.\n\n" + "\n\n".join(partes) + "\n"


def para_latex(df: pd.DataFrame) -> str:
    """Tabela no padrão booktabs, agrupada por família (cole no artigo em LaTeX/SBC)."""
    cab = " & ".join(COLUNAS.values())
    linhas = [r"\begin{table*}[t]", r"\centering\small",
              r"\caption{Desempenho no conjunto de teste (1.080 notícias). Precisão, revocação e F1 da classe falsa, em \%. "
              r"IC 95\%: bootstrap percentil sobre pares (2.000 reamostras).}",
              r"\label{tab:resultados}", r"\begin{tabular}{l" + "c" * (len(COLUNAS) - 1) + "}", r"\toprule",
              cab + r" \\", r"\midrule"]
    for familia, sub in df.groupby("familia", sort=False):
        linhas.append(rf"\multicolumn{{{len(COLUNAS)}}}{{l}}{{\textit{{{familia}}}}} \\")
        for _, r in sub.iterrows():
            linhas.append(" & ".join(str(r[c]).replace("%", r"\%").replace("±", r"$\pm$") for c in COLUNAS) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(linhas) + "\n"


# --------------------------------------------------------------------------------------------
# Testes pareados
# --------------------------------------------------------------------------------------------
def holm(p_valores: list[float]) -> list[float]:
    """Correção de Holm-Bonferroni para comparações múltiplas (p ajustados, monotônicos)."""
    m = len(p_valores)
    ordem = np.argsort(p_valores)
    ajustados, maximo = np.empty(m), 0.0
    for posicao, i in enumerate(ordem):
        maximo = max(maximo, min(1.0, (m - posicao) * p_valores[i]))
        ajustados[i] = maximo
    return ajustados.tolist()


def tabela_mcnemar(modelos: list[str], registros: dict[str, dict]) -> pd.DataFrame:
    acertos = {m: (lambda d: (d.rotulo == d.predicao).to_numpy())(carregar_predicoes(m)) for m in modelos}
    linhas = []
    for a, b in combinations(modelos, 2):
        r = mcnemar(acertos[a], acertos[b])
        linhas.append({"modelo_A": a, "modelo_B": b, "acc_A": _pct(acertos[a].mean()), "acc_B": _pct(acertos[b].mean()),
                       "A_acerta_B_erra": r["b"], "A_erra_B_acerta": r["c"], "p": r["p_valor"]})
    df = pd.DataFrame(linhas)
    if len(df):
        df["p_holm"] = holm(df["p"].tolist())
        df["significativo_5%"] = np.where(df["p_holm"] < 0.05, "sim", "não")
        df["p"] = df["p"].map(_p)
        df["p_holm"] = df["p_holm"].map(_p)
    return df


# --------------------------------------------------------------------------------------------
# Robustez aos artefatos e custo
# --------------------------------------------------------------------------------------------
def execucoes(nome: str, registros: dict[str, dict]) -> tuple[list[float], str | None]:
    """Acurácias de teste de um modelo (uma) ou de um grupo de sementes (várias) e o representante.

    O representante (usado no McNemar) é a semente 42 quando o nome é de um grupo.
    """
    if nome in registros and "teste" in registros[nome]:
        return [registros[nome]["teste"]["acuracia"]], nome
    membros = sorted(m for m, r in registros.items() if r["info"].get("grupo") == nome and "teste" in r)
    if not membros:
        return [], None
    rep = f"{nome}_s42" if f"{nome}_s42" in membros else membros[0]
    return [registros[m]["teste"]["acuracia"] for m in membros], rep


def _media_dp(v: list[float]) -> str:
    return _pct(np.mean(v)) + (f" ± {_pct(np.std(v, ddof=1))}" if len(v) > 1 else "")


def _nome_curto(registro: dict, padrao: str) -> str:
    return registro["info"].get("descricao", padrao).split(" (")[0]


def tabela_robustez(registros: dict[str, dict]) -> pd.DataFrame:
    linhas = []
    for minimo, neutro in PARES_NEUTRO:
        (acc_m, rep_m), (acc_n, rep_n) = execucoes(minimo, registros), execucoes(neutro, registros)
        if not acc_m or not acc_n:
            continue
        a, b = carregar_predicoes(rep_m), carregar_predicoes(rep_n)
        teste = mcnemar((a.rotulo == a.predicao).to_numpy(), (b.rotulo == b.predicao).to_numpy())
        linhas.append({"modelo": _nome_curto(registros[rep_m], minimo),
                       "texto_minimo": _media_dp(acc_m), "texto_neutralizado": _media_dp(acc_n),
                       "diferenca_pp": f"{100 * (np.mean(acc_n) - np.mean(acc_m)):+.1f}".replace(".", ","),
                       "p": teste["p_valor"]})
    df = pd.DataFrame(linhas)
    if len(df):
        df["p_holm"] = [_p(x) for x in holm(df["p"].tolist())]
        df["p"] = df["p"].map(_p)
    return df


def tabela_custo(registros: dict[str, dict]) -> pd.DataFrame:
    """Tempo de ajuste (busca de hiperparâmetros ou treino) e de inferência no teste, por modelo."""
    linhas: dict[str, list] = {}
    for modelo, reg in registros.items():
        info = reg["info"]
        if "teste" not in reg or info.get("versao_corpus") == "completo" or modelo.endswith("_calibrado"):
            continue   # a versão calibrada do LLM tem o mesmo custo da versão com limiar 0
        ajuste = info.get("tempo_treino_s", info.get("tempo_busca_s", 0)) + info.get("tempo_codificacao_s", 0)
        inferencia = info.get("tempo_inferencia_teste_s")
        if inferencia is None and isinstance(info.get("tempo_inferencia_s"), dict):
            inferencia = info["tempo_inferencia_s"].get("teste")
        if inferencia is None and "tempo_codificacao_s" in info:
            # embeddings: a inferência é dominada pela codificação, medida no corpus inteiro (7.200)
            inferencia = info["tempo_codificacao_s"] * reg["teste"]["n"] / 7200
        linhas.setdefault(info.get("grupo", modelo), []).append(
            (info.get("familia", ""), info.get("descricao", modelo), ajuste, inferencia, "modelo_hf" in info))
    tabela = []
    for regs in linhas.values():
        familia, desc, _, _, gpu = regs[0]
        inf = [r[3] for r in regs if r[3] is not None]
        ajuste = float(np.mean([r[2] for r in regs]))
        inferencia = float(np.mean(inf)) if inf else None
        tabela.append({"familia": familia, "modelo": desc, "_ordem": ajuste, "ajuste_s": _num(ajuste),
                       "inferencia_teste_s": _num(inferencia),
                       "ms_por_noticia": _num(1000 * inferencia / 1080 if inferencia else None, 2),
                       "hardware": "GPU" if gpu else "CPU (20 threads)"})
    df = pd.DataFrame(tabela).sort_values(["familia", "_ordem"]).reset_index(drop=True)
    return df.drop(columns="_ordem")


# --------------------------------------------------------------------------------------------
# Análise de erros
# --------------------------------------------------------------------------------------------
def acuracia_por(modelos: list[str], coluna: str, filtro=None) -> pd.DataFrame:
    tabelas = {}
    for m in modelos:
        df = carregar_predicoes(m)
        if filtro is not None:
            df = filtro(df)
        df["acerto"] = df.rotulo == df.predicao
        tabelas[m] = df.groupby(coluna)["acerto"].mean().map(_pct)
        n = df.groupby(coluna).size()
    return pd.DataFrame(tabelas).assign(n=n).sort_values("n", ascending=False)


def acuracia_por_final(modelos: list[str]) -> pd.DataFrame:
    """Acurácia por classe e por "o texto original termina com pontuação final".

    A normalização de tamanho trunca a notícia mais longa do par no meio da frase. O caso
    típico é falsa completa (com ponto final) e verdadeira truncada (sem). Os subgrupos
    contrários (falsa truncada, verdadeira completa) mostram se o modelo depende desse atalho.
    """
    import re
    from . import dados

    termina = {d.id: bool(re.search(r"[.!?\"”']\s*$", d.texto)) for d in dados.carregar("normalizado")}
    tabelas, n = {}, None
    for m in modelos:
        df = carregar_predicoes(m)
        df["final"] = [("com" if termina[i] else "sem") + " pontuação final" for i in df.id]
        df["classe"] = df.rotulo.map({config.POSITIVO: "Falsa", config.NEGATIVO: "Verdadeira"})
        df["acerto"] = df.rotulo == df.predicao
        tabelas[m] = df.groupby(["classe", "final"])["acerto"].mean().map(_pct)
        n = df.groupby(["classe", "final"]).size()
    return pd.DataFrame(tabelas).assign(n=n)


def erros_comuns(modelos: list[str]) -> pd.DataFrame:
    """Notícias de teste erradas por todos os modelos principais, com um trecho do texto."""
    from . import dados

    base = carregar_predicoes(modelos[0])[["id", "par", "categoria", "fonte", "palavras", "rotulo"]]
    erra = np.ones(len(base), bool)
    for m in modelos:
        p = carregar_predicoes(m)
        erra &= (p.rotulo != p.predicao).to_numpy()
    comuns = base[erra].copy()
    textos = {d.id: d.texto for d in dados.carregar("normalizado")}
    comuns["trecho"] = comuns["id"].map(lambda i: textos[i][:300].replace("\n", " "))
    return comuns


# --------------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------------
def figura_acuracia(registros: dict[str, dict], incluir: list[str] | None = None,
                    nome: str = "comparacao_acuracia") -> None:
    """Acurácia no teste com IC95% (ou média ± dp nas sementes), um ponto por modelo, por família.

    `incluir`: nomes de modelos ou de grupos de sementes a mostrar (None = todos). As cores das
    famílias são as mesmas em todas as versões da figura.
    """
    import matplotlib.pyplot as plt
    estilo.aplicar()
    todas_familias = sorted({r["info"].get("familia", "") for r in registros.values() if "teste" in r})
    itens, grupos = [], {}
    for modelo, reg in registros.items():
        if "teste" not in reg:
            continue
        info = reg["info"]
        if incluir is not None and modelo not in incluir and info.get("grupo") not in incluir:
            continue
        if "grupo" in info:
            grupos.setdefault(info["grupo"], []).append(reg)
            continue
        m = reg["teste"]
        itens.append((info.get("familia", ""), info.get("descricao", modelo), m["acuracia"], *m["ic95"]["acuracia"]))
    for grupo, regs in grupos.items():
        v = np.array([r["teste"]["acuracia"] for r in regs])
        dp = v.std(ddof=1) if len(v) > 1 else 0
        itens.append((regs[0]["info"]["familia"], f"{regs[0]['info']['descricao']} (média de {len(v)})",
                      v.mean(), v.mean() - dp, v.mean() + dp))
    itens.sort(key=lambda x: (x[0], -x[2]), reverse=True)

    familias = sorted({i[0] for i in itens})
    cor = {f: estilo.CATEGORICAS[k % len(estilo.CATEGORICAS)] for k, f in enumerate(todas_familias)}
    fig, eixo = plt.subplots(figsize=(7, 0.32 * len(itens) + 0.8))
    for y, (fam, desc, acc, lo, hi) in enumerate(itens):
        eixo.plot([lo, hi], [y, y], color=cor[fam], linewidth=2, solid_capstyle="round")
        eixo.plot(acc, y, "o", color=cor[fam], markersize=7, markeredgecolor=estilo.SUPERFICIE, markeredgewidth=1.5)
        eixo.text(hi + 0.004, y, _pct(acc), va="center", fontsize=8, color=estilo.TINTA_SECUNDARIA)
    eixo.set_yticks(range(len(itens)), [i[1] for i in itens])
    eixo.axvline(0.5, color=estilo.TINTA_SECUNDARIA, linewidth=0.8, linestyle=":")
    eixo.set_xlabel("Acurácia no teste, % (IC 95% por bootstrap de pares; BERTimbau: média ± dp entre sementes)")
    eixo.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
    eixo.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    eixo.legend([Line2D([], [], color=cor[f], marker="o", linewidth=2) for f in familias], familias,
                loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    estilo.salvar(fig, nome, config.FIGURAS)
    plt.close(fig)


def figura_robustez(registros: dict[str, dict]) -> None:
    """Cada modelo no texto mínimo e no neutralizado, ligados por um segmento (gráfico de halteres)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    estilo.aplicar()
    itens = []
    for minimo, neutro in PARES_NEUTRO:
        (acc_m, rep_m), (acc_n, _) = execucoes(minimo, registros), execucoes(neutro, registros)
        if acc_m and acc_n:
            itens.append((_nome_curto(registros[rep_m], minimo).split(",")[0], np.mean(acc_m), np.mean(acc_n)))
    if not itens:
        return
    cor_m, cor_n = estilo.CATEGORICAS[0], estilo.CATEGORICAS[1]
    fig, eixo = plt.subplots(figsize=(7, 0.42 * len(itens) + 0.9))
    for y, (desc, m, n) in enumerate(reversed(itens)):
        eixo.plot([m, n], [y, y], color=estilo.GRADE, linewidth=3, solid_capstyle="round", zorder=1)
        for x, cor in ((m, cor_m), (n, cor_n)):
            eixo.plot(x, y, "o", color=cor, markersize=8, markeredgecolor=estilo.SUPERFICIE,
                      markeredgewidth=1.5, zorder=2)
        eixo.text(max(m, n) + 0.012, y, f"{100 * (n - m):+.1f} pp".replace(".", ","), va="center",
                  fontsize=8, color=estilo.TINTA_SECUNDARIA)
    eixo.set_yticks(range(len(itens)), [i[0] for i in reversed(itens)])
    eixo.axvline(0.5, color=estilo.TINTA_SECUNDARIA, linewidth=0.8, linestyle=":")
    eixo.set_xlim(0.45, 1.06)
    eixo.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
    eixo.set_xlabel("Acurácia no teste, %")
    eixo.grid(axis="y", visible=False)
    eixo.legend([Line2D([], [], color=c, marker="o", linestyle="") for c in (cor_m, cor_n)],
                ["Texto mínimo", "Artefatos de formatação neutralizados"],
                loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    estilo.salvar(fig, "robustez_artefatos", config.FIGURAS)
    plt.close(fig)


def figura_matrizes(modelos: list[str], registros: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    estilo.aplicar()
    mapa = LinearSegmentedColormap.from_list("seq", estilo.SEQUENCIAL)
    n = len(modelos)
    colunas = min(n, 4)
    linhas = int(np.ceil(n / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(1.9 * colunas, 2.0 * linhas), squeeze=False)
    rotulos = [estilo.NOME_CLASSE[r] for r in config.ROTULOS]
    for eixo, m in zip(eixos.flat, modelos):
        mc = np.array(registros[m]["teste"]["matriz_confusao"])
        eixo.imshow(mc, cmap=mapa, vmin=0, vmax=mc.sum() / 2)
        for (i, j), v in np.ndenumerate(mc):
            escuro = v > mc.sum() / 4
            eixo.text(j, i, str(v), ha="center", va="center", fontsize=9,
                      color=estilo.SUPERFICIE if escuro else estilo.TINTA)
        eixo.set_title(registros[m]["info"].get("descricao", m)[:34], fontsize=8, loc="left")
        eixo.set_xticks([0, 1], rotulos, fontsize=7)
        eixo.set_yticks([0, 1], rotulos, fontsize=7)
        eixo.grid(False)
        for s in eixo.spines.values():
            s.set_visible(False)
    for eixo in list(eixos.flat)[n:]:
        eixo.axis("off")
    fig.supxlabel("Classe predita", fontsize=8)
    fig.supylabel("Classe real", fontsize=8)
    fig.tight_layout()
    estilo.salvar(fig, "matrizes_confusao", config.FIGURAS)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def main() -> None:
    config.garantir_pastas()
    registros = carregar_metricas()
    if not registros:
        raise SystemExit("nenhum resultado em resultados/metricas; rode os experimentos antes.")
    escrever = lambda nome, texto: (config.TABELAS / nome).write_text(texto, encoding="utf-8")

    teste = tabela_resultados(registros, "teste")
    escrever("resultados_teste.md", para_markdown(teste))
    escrever("resultados_teste.tex", para_latex(teste))
    escrever("resultados_validacao.md", para_markdown(tabela_resultados(registros, "validacao")))
    print(para_markdown(teste))

    principais = [m for m in PRINCIPAIS if m in registros and "teste" in registros[m]]
    neutros = [m for m in NEUTROS if m in registros and "teste" in registros[m]]
    print("modelos principais encontrados:", principais, "| neutralizados:", neutros)

    escrever("robustez_artefatos.md", "Acurácia no teste (%) no texto mínimo e com os artefatos de formatação "
             "neutralizados; McNemar entre as duas versões (semente 42 nos modelos com várias sementes), "
             "p ajustado por Holm.\n\n" + tabela_robustez(registros).to_markdown(index=False) + "\n")
    escrever("custo.md", "Tempo de ajuste (busca de hiperparâmetros com validação cruzada 5x, ou treino; nos "
             "embeddings inclui codificar o corpus inteiro) e de inferência nas 1.080 notícias de teste, em segundos. "
             "GPU: NVIDIA RTX 2000 Ada 16 GB. Tempos aproximados: alguns experimentos dividiram a GPU com outro.\n\n"
             + tabela_custo(registros).to_markdown(index=False) + "\n")
    mc = tabela_mcnemar(principais, registros)
    escrever("mcnemar.md", "Teste exato de McNemar no conjunto de teste; p ajustado por Holm.\n\n"
             + mc.to_markdown(index=False) + "\n")

    analisados = principais + neutros
    escrever("erros_por_categoria.md", acuracia_por(analisados, "categoria").to_markdown() + "\n")
    so_falsas = lambda df: df[df.rotulo == config.POSITIVO].assign(
        origem=lambda d: np.where(d.fonte == SITE_DOMINANTE, SITE_DOMINANTE, "outras fontes"))
    escrever("erros_por_fonte.md", "Revocação da classe falsa conforme a fonte da notícia falsa.\n\n"
             + acuracia_por(analisados, "origem", so_falsas).to_markdown() + "\n")
    escrever("erros_por_final.md", "Acurácia (%) por classe e por como o texto original termina. Subgrupos "
             "típicos: falsa com pontuação final, verdadeira sem (truncada). Contrários ao atalho: falsa sem, "
             "verdadeira com.\n\n" + acuracia_por_final(analisados).to_markdown() + "\n")
    por_tamanho = lambda df: df.assign(quartil_tamanho=pd.qcut(df.palavras, 4, labels=["Q1 (curtas)", "Q2", "Q3", "Q4 (longas)"]))
    escrever("erros_por_tamanho.md", acuracia_por(analisados, "quartil_tamanho", por_tamanho).sort_index().to_markdown() + "\n")
    if principais:
        comuns = erros_comuns(principais)
        comuns.to_csv(config.TABELAS / "erros_comuns.csv", index=False, encoding="utf-8")
        print(f"{len(comuns)} notícias de teste erradas por todos os {len(principais)} modelos principais")

    termos = config.TABELAS / "termos_tfidf_svm.json"
    if termos.exists():
        t = json.loads(termos.read_text(encoding="utf-8"))
        df = pd.DataFrame({"Falsa": [f"{w} ({p:+.2f})".replace(".", ",") for w, p in t[config.POSITIVO][:20]],
                           "Verdadeira": [f"{w} ({p:+.2f})".replace(".", ",") for w, p in t[config.NEGATIVO][:20]]})
        escrever("termos_tfidf_svm.md", "Termos de maior peso na SVM linear (TF-IDF 1-2, sem stopwords).\n\n"
                 + df.to_markdown(index=False) + "\n")

    figura_acuracia(registros)
    figura_acuracia(registros, FIGURA_ARTIGO, "comparacao_acuracia_artigo")
    figura_robustez(registros)
    if principais:
        figura_matrizes(principais, registros)
    print("tabelas em", config.TABELAS, "\nfiguras em", config.FIGURAS)


if __name__ == "__main__":
    main()
