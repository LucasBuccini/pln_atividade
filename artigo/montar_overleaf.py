"""Monta a entrega do Overleaf: copia as figuras, confere o .tex e gera artigo/entrega_overleaf.zip.

    python artigo/montar_overleaf.py

O artigo usa o modelo da Linguamática (linguamatica.sty + sp_por.bst), que já está no projeto
do template no Overleaf. O zip leva main.tex, referencias.bib e figuras/: no projeto do template,
use "Upload" para enviar esses arquivos, substituindo o main.tex de exemplo (compilador pdfLaTeX).

Sem LaTeX instalado aqui, a conferência é estática: figuras citadas existem, toda chave de
\\cite está no .bib, todo \\ref tem \\label, ambientes \\begin/\\end e chaves estão balanceados.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ARTIGO = Path(__file__).resolve().parent
PROJETO = ARTIGO / "overleaf"
FIGURAS = ARTIGO.parent / "resultados" / "figuras"
ZIP = ARTIGO / "entrega_overleaf.zip"


def copiar_figuras(tex: str) -> None:
    destino = PROJETO / "figuras"
    destino.mkdir(exist_ok=True)
    for nome in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figuras/([^}]+)\}", tex):
        shutil.copy2(FIGURAS / nome, destino / nome)


def conferir(tex: str) -> list[str]:
    problemas = []
    sem_comentarios = re.sub(r"(?<!\\)%.*", "", tex)

    for nome in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", sem_comentarios):
        if not (PROJETO / nome).exists():
            problemas.append(f"figura inexistente: {nome}")

    bib = (PROJETO / "referencias.bib").read_text(encoding="utf-8")
    chaves_bib = set(re.findall(r"@\w+\{([^,]+),", bib))
    for grupo in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", sem_comentarios):
        for chave in grupo.split(","):
            if chave.strip() not in chaves_bib:
                problemas.append(f"citação sem entrada no .bib: {chave.strip()}")

    rotulos = set(re.findall(r"\\label\{([^}]+)\}", sem_comentarios))
    for ref in re.findall(r"\\ref\{([^}]+)\}", sem_comentarios):
        if ref not in rotulos:
            problemas.append(f"\\ref sem \\label: {ref}")

    abertos = Counter(re.findall(r"\\begin\{([^}]+)\}", sem_comentarios))
    fechados = Counter(re.findall(r"\\end\{([^}]+)\}", sem_comentarios))
    for amb in abertos | fechados:
        if abertos[amb] != fechados[amb]:
            problemas.append(f"ambiente desbalanceado: {amb} ({abertos[amb]} begin, {fechados[amb]} end)")

    sem_escapes = re.sub(r"\\[{}]", "", sem_comentarios)
    if sem_escapes.count("{") != sem_escapes.count("}"):
        problemas.append(f"chaves desbalanceadas: {sem_escapes.count('{')} '{{' e {sem_escapes.count('}')} '}}'")

    if re.search(r"(?<!\\)[%](?=\d)", sem_comentarios):
        problemas.append("possível % sem escape antes de número")
    for car in "“”‘’−×≥≤…":
        if car in sem_comentarios:
            problemas.append(f"caractere Unicode que o pdfLaTeX pode recusar: {car!r}")

    a_conferir = len(re.findall(r"\\conferir\{", sem_comentarios))
    if a_conferir > 0:
        problemas.append(f"aviso: {a_conferir} trechos \\conferir{{}} (dados de terceiros) a confirmar na fonte")
    return problemas


def main() -> None:
    tex = (PROJETO / "main.tex").read_text(encoding="utf-8")
    copiar_figuras(tex)
    problemas = conferir(tex)
    for p in problemas:
        print(" -", p)
    erros = [p for p in problemas if not p.startswith("aviso")]
    if erros:
        sys.exit(f"{len(erros)} problema(s); zip não gerado.")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for arq in sorted(PROJETO.rglob("*")):
            if arq.is_file():
                z.write(arq, arq.relative_to(PROJETO))
    print(f"ok: {ZIP} ({ZIP.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
