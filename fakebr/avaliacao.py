"""Avaliação comum a todos os modelos: métricas, intervalos de confiança, McNemar e registro.

Todo experimento termina chamando `registrar(...)`, que grava:

- `resultados/predicoes/<modelo>_<parte>.csv`: id, par, categoria, rótulo, predição e escore
  de cada notícia, para a análise de erros e para os testes pareados entre modelos;
- `resultados/metricas/<modelo>.json`: métricas por parte, intervalos de confiança e as
  informações do experimento (hiperparâmetros, tempo, pré-processamento).

Convenção: o escore é maior quanto mais o modelo acredita que a notícia é falsa.
A classe positiva é "fake" (config.POSITIVO).
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from . import config


def metricas(y_true: Sequence, y_pred: Sequence, escores: Sequence | None = None) -> dict[str, Any]:
    """Acurácia, precisão/revocação/F1 da classe falsa, F1 macro, MCC, ROC-AUC e matriz de confusão."""
    pos = config.POSITIVO
    res = {
        "n": len(y_true),
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao_fake": precision_score(y_true, y_pred, pos_label=pos, zero_division=0),
        "revocacao_fake": recall_score(y_true, y_pred, pos_label=pos, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, pos_label=pos, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        # Linhas = verdadeiro, colunas = predito, na ordem (true, fake): [[VN, FP], [FN, VP]]
        "matriz_confusao": confusion_matrix(y_true, y_pred, labels=list(config.ROTULOS)).tolist(),
    }
    if escores is not None and len(set(y_true)) == 2:
        res["roc_auc"] = roc_auc_score([y == pos for y in y_true], escores)
    return {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}


def bootstrap_ic(y_true: Sequence, y_pred: Sequence, grupos: Sequence, n_reamostras: int = 2000,
                 confianca: float = 0.95, semente: int = config.SEMENTE) -> dict[str, list[float]]:
    """Intervalo de confiança por bootstrap percentil, reamostrando grupos (pares) com reposição.

    As duas notícias de um par não são independentes; reamostrar notícias isoladas
    subestimaria a variância. Por isso a unidade reamostrada é o par.
    """
    real = np.asarray(y_true) == config.POSITIVO
    pred = np.asarray(y_pred) == config.POSITIVO
    _, indice = np.unique(np.asarray(grupos), return_inverse=True)
    n_grupos = indice.max() + 1
    # Contagens da matriz de confusão por grupo; uma reamostra é só a soma das contagens sorteadas.
    contagens = np.zeros((n_grupos, 4))
    for k, celula in enumerate((real & pred, ~real & pred, real & ~pred, ~real & ~pred)):
        contagens[:, k] = np.bincount(indice, weights=celula, minlength=n_grupos)
    rng = np.random.default_rng(semente)
    pesos = np.stack([np.bincount(rng.integers(0, n_grupos, n_grupos), minlength=n_grupos)
                      for _ in range(n_reamostras)])
    vp, fp, fn, vn = (pesos @ contagens).T
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_pos = np.nan_to_num(2 * vp / (2 * vp + fp + fn))
        f1_neg = np.nan_to_num(2 * vn / (2 * vn + fn + fp))
    amostras = {"acuracia": (vp + vn) / (vp + fp + fn + vn), "f1_fake": f1_pos, "f1_macro": (f1_pos + f1_neg) / 2}
    alfa = (1 - confianca) / 2 * 100
    return {m: [round(float(np.percentile(v, alfa)), 6), round(float(np.percentile(v, 100 - alfa)), 6)]
            for m, v in amostras.items()}


def mcnemar(acertos_a: Sequence[bool], acertos_b: Sequence[bool]) -> dict[str, float]:
    """Teste exato de McNemar entre dois classificadores avaliados nas mesmas notícias.

    Considera só as discordâncias: b = A acerta e B erra, c = A erra e B acerta.
    Sob a hipótese nula de desempenho igual, b ~ Binomial(b + c, 0.5) (Dietterich, 1998).
    """
    a, b_ = np.asarray(acertos_a, bool), np.asarray(acertos_b, bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"b": b, "c": c, "p_valor": float(p)}


def registrar(modelo: str, parte: str, corpus, y_pred: Sequence, escores: Sequence | None = None,
              info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Grava predições e métricas de um modelo numa parte (validacao/teste) e devolve as métricas.

    `modelo`: nome curto e único, usado nos arquivos e nas tabelas (ex.: "tfidf_svm").
    `corpus`: o Corpus avaliado, na mesma ordem de `y_pred`.
    `info`:   o que for preciso para reproduzir (hiperparâmetros, tempo, pré-processamento).
              Chaves "familia" e "descricao" aparecem nas tabelas do artigo.
    """
    import pandas as pd

    config.garantir_pastas()
    y_true = corpus.rotulos
    res = metricas(y_true, y_pred, escores)
    res["ic95"] = bootstrap_ic(y_true, y_pred, [d.metadados["par"] for d in corpus])

    pd.DataFrame({
        "id": corpus.ids,
        "par": [d.metadados["par"] for d in corpus],
        "categoria": [d.metadados["categoria"] for d in corpus],
        "fonte": [d.metadados["fonte"] for d in corpus],
        "palavras": [d.metadados["palavras_original"] for d in corpus],
        "rotulo": y_true,
        "predicao": list(y_pred),
        "escore": list(escores) if escores is not None else None,
    }).to_csv(config.PREDICOES / f"{modelo}_{parte}.csv", index=False, encoding="utf-8")

    caminho = config.METRICAS / f"{modelo}.json"
    registro = json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}
    registro["modelo"] = modelo
    registro.setdefault("info", {}).update(info or {})
    registro["info"]["atualizado_em"] = time.strftime("%Y-%m-%d %H:%M:%S")
    registro[parte] = res
    caminho.write_text(json.dumps(registro, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    ic = res["ic95"]["acuracia"]
    print(f"[{modelo} | {parte}] acurácia {res['acuracia']:.4f} (IC95% {ic[0]:.4f}–{ic[1]:.4f}) "
          f"F1(fake) {res['f1_fake']:.4f}" + (f" AUC {res['roc_auc']:.4f}" if "roc_auc" in res else ""))
    return res


class Cronometro:
    """Mede o tempo de um bloco: `with Cronometro() as t: ...` e depois `t.segundos`."""

    def __enter__(self):
        self.inicio = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.segundos = round(time.perf_counter() - self.inicio, 2)
