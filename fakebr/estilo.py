"""Estilo único das figuras do artigo (matplotlib), para que todas leiam como um conjunto.

Cores categóricas em ordem fixa (paleta validada para daltonismo nos pares adjacentes):
a classe "true" é sempre o azul e a classe "fake" sempre o laranja, em todas as figuras.
Texto usa tinta neutra, nunca a cor da série; grade e eixos ficam discretos.
"""
from __future__ import annotations

from pathlib import Path

CATEGORICAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COR_CLASSE = {"true": CATEGORICAS[0], "fake": CATEGORICAS[1]}
NOME_CLASSE = {"true": "Verdadeira", "fake": "Falsa"}
SEQUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRADE = "#e4e3df"
SUPERFICIE = "#ffffff"   # branco: as figuras vão para um PDF impresso


def aplicar() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "text.color": TINTA, "axes.labelcolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
        "axes.edgecolor": GRADE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAS),
        "lines.linewidth": 2, "legend.frameon": False,
    })


def salvar(fig, nome: str, pasta: Path) -> None:
    """Grava a figura em PDF (para o LaTeX) e PNG (para o Markdown/apresentação)."""
    pasta.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(pasta / f"{nome}.{ext}")
