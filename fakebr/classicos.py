"""Experimentos 0 e 1: viés de tamanho e linha de base estatística (TF-IDF + NB / SVM / Regressão Logística).

    python -m fakebr.classicos                      # roda todos os experimentos
    python -m fakebr.classicos tfidf_svm tfidf_nb   # roda só os indicados
    python -m fakebr.classicos --listar             # lista os experimentos disponíveis

Protocolo (seção "Metodologia" do artigo):

- Hiperparâmetros escolhidos por GridSearchCV com validação cruzada de 5 dobras no treino,
  agrupada por par (GroupKFold), para que as duas notícias de um par nunca fiquem uma no
  treino e outra na validação da mesma dobra.
- O melhor modelo é retreinado no treino inteiro e avaliado uma vez na validação e no teste.
- O pré-processamento não aprende nada dos dados (só regras fixas), então é calculado uma
  vez para o corpus inteiro e guardado em cache, sem risco de vazamento. O TF-IDF, que
  aprende vocabulário e IDF, fica dentro do pipeline do scikit-learn e é ajustado por dobra.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline as PipelineSk
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from pln_prep import (NormalizarEspacos, ParaMinusculas, Pipeline, RemoverEmails, RemoverHTML, RemoverNumeros,
                      RemoverPontuacao, RemoverTokensCurtos, RemoverURLs, SubstituirPadrao, TokenizarPorRegex,
                      pipelines_prontos)

from . import config, dados
from .avaliacao import Cronometro, registrar


# --------------------------------------------------------------------------------------------
# Pré-processamentos comparados (todos da biblioteca pln_prep)
# --------------------------------------------------------------------------------------------
def _sem_remocao_de_stopwords() -> Pipeline:
    """Mesma limpeza de para_modelos_classicos(), mas mantendo as palavras funcionais."""
    return Pipeline([RemoverHTML(), RemoverURLs(), RemoverEmails(), ParaMinusculas(), RemoverNumeros(),
                     RemoverPontuacao(), NormalizarEspacos(), TokenizarPorRegex(), RemoverTokensCurtos(2)])


def _neutro() -> Pipeline:
    """Limpeza mínima + remoção dos artefatos de formatação que denunciam a classe (Seção 3 do artigo).

    - aspas curvas e retas viram a mesma aspa ("); apóstrofos curvos viram ';
    - o ". ." que separa título e corpo em parte das notícias vira um ponto só;
    - o fim do texto é padronizado: a normalização de tamanho trunca a notícia mais longa do
      par (quase sempre a verdadeira) no meio de uma frase, então terminar sem ponto final
      virou quase um rótulo (70% das falsas terminam com pontuação, 7% das verdadeiras).
      Todo texto é cortado na última frase completa e perde a pontuação final, de modo que
      nenhum, de nenhuma classe, termina em frase interrompida ou com marca de fim.
    """
    return Pipeline([*pipelines_prontos.para_transformers().etapas,
                     SubstituirPadrao(r"[“”„«»]", '"'), SubstituirPadrao(r"[‘’]", "'"),
                     SubstituirPadrao(r"\.(\s+\.)+", "."),
                     SubstituirPadrao(r"(?<=[.!?…])[^.!?…]*$", ""),          # descarta a frase final incompleta
                     SubstituirPadrao(r"[\s\"'.!?…:;,)\]-]+$", "")])         # e a pontuação que sobrar no fim


PREPROCESSAMENTOS: dict[str, Callable[[], Pipeline]] = {
    "sem_stopwords": lambda: pipelines_prontos.para_modelos_classicos("portugues"),
    "com_stopwords": _sem_remocao_de_stopwords,
    "radical": lambda: pipelines_prontos.para_modelos_classicos("portugues", normalizacao="radical"),
    "lema": lambda: pipelines_prontos.para_modelos_classicos("portugues", normalizacao="lema"),
    # Limpeza mínima: mantém caixa, pontuação e números (sinal estilístico para n-gramas de caracteres).
    "minimo": pipelines_prontos.para_transformers,
    "neutro": _neutro,
}


def textos_preprocessados(corpus, versao: str, variante: str) -> list[str]:
    """Aplica o pré-processamento `variante` ao corpus, usando cache em disco (dados/cache)."""
    config.garantir_pastas()
    caminho = config.CACHE / f"{versao}_{variante}.json"
    pipeline = PREPROCESSAMENTOS[variante]()
    # O cache só vale se foi gerado com exatamente a mesma configuração de pipeline.
    if caminho.exists():
        cache = json.loads(caminho.read_text(encoding="utf-8"))
        if cache.get("pipeline") == pipeline.para_config() and all(i in cache["textos"] for i in corpus.ids):
            return [cache["textos"][i] for i in corpus.ids]
    with Cronometro() as t:
        textos = pipeline.transformar(corpus)
    print(f"  pré-processamento {variante!r} ({versao}): {t.segundos}s")
    cache = {"pipeline": pipeline.para_config(), "textos": dict(zip(corpus.ids, textos))}
    caminho.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return textos


def assinatura_preprocessamento(variante: str) -> str:
    """Hash curto da configuração do pipeline, para invalidar caches derivados (ex.: embeddings)."""
    import hashlib
    texto = json.dumps(PREPROCESSAMENTOS[variante]().para_config(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------------------------
# Atributos superficiais (Experimento 0): o que um modelo acerta sem ler o conteúdo
# --------------------------------------------------------------------------------------------
def _atributos_de_comprimento(textos) -> np.ndarray:
    """log(1 + nº de palavras) e log(1 + nº de caracteres) de cada texto."""
    return np.array([[np.log1p(len(t.split())), np.log1p(len(t))] for t in textos])


_MARCAS_DE_FORMATACAO = [
    re.compile(r"[.!?\"”']\s*$"),   # termina com pontuação final
    re.compile(r"\""),              # usa aspas retas
    re.compile(r"[“”]"),            # usa aspas curvas
    re.compile(r"\.\s+\."),         # ". ." entre título e corpo
]


def _atributos_de_formatacao(textos) -> np.ndarray:
    """Presença (0/1) de cada marca de formatação que difere entre as classes."""
    return np.array([[float(bool(p.search(t))) for p in _MARCAS_DE_FORMATACAO] for t in textos])


# --------------------------------------------------------------------------------------------
# Definição declarativa dos experimentos
# --------------------------------------------------------------------------------------------
def _tfidf(**kw) -> TfidfVectorizer:
    # sublinear_tf: log(1 + tf), atenua palavras repetidas muitas vezes no mesmo texto.
    # min_df=2: descarta termos de um documento só (ruído, nomes raros) e reduz a dimensionalidade.
    padrao = {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "sublinear_tf": True, "lowercase": False}
    return TfidfVectorizer(**{**padrao, **kw})


@dataclass
class Experimento:
    nome: str
    descricao: str
    familia: str
    preprocessamento: str
    criar: Callable[[], PipelineSk]
    grade: dict[str, list[Any]] = field(default_factory=dict)
    versao: str = "normalizado"


C_SVM = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C_REGLOG = [1, 3, 10, 30, 100]
ALFA_NB = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]

EXPERIMENTOS = [
    # Experimento 0: classificadores que não leem o conteúdo, só tamanho ou formatação.
    Experimento("comprimento_completo", "RegLog só com o tamanho do texto (textos completos)",
                "0. Atalhos", "minimo",
                lambda: make_pipeline(FunctionTransformer(_atributos_de_comprimento), StandardScaler(),
                                      LogisticRegression()), versao="completo"),
    Experimento("comprimento", "RegLog só com o tamanho do texto (textos normalizados)",
                "0. Atalhos", "minimo",
                lambda: make_pipeline(FunctionTransformer(_atributos_de_comprimento), StandardScaler(),
                                      LogisticRegression())),
    Experimento("tfidf_svm_completo", "TF-IDF (1-2) + SVM linear nos textos completos",
                "0. Atalhos", "sem_stopwords",
                lambda: make_pipeline(_tfidf(), LinearSVC()), {"linearsvc__C": C_SVM}, versao="completo"),
    Experimento("formatacao", "RegLog só com 4 marcas de formatação (pontuação final, aspas, '. .')",
                "0. Atalhos", "minimo",
                lambda: make_pipeline(FunctionTransformer(_atributos_de_formatacao), LogisticRegression())),
    # Controles: depois da neutralização as marcas não devem mais separar as classes,
    # e o corte na última frase completa não pode criar um novo atalho de tamanho.
    Experimento("comprimento_neutro", "RegLog só com o tamanho do texto, após neutralizar o fim dos textos",
                "0. Atalhos", "neutro",
                lambda: make_pipeline(FunctionTransformer(_atributos_de_comprimento), StandardScaler(),
                                      LogisticRegression())),
    Experimento("formatacao_neutro", "RegLog só com as marcas de formatação, após neutralizá-las",
                "0. Atalhos", "neutro",
                lambda: make_pipeline(FunctionTransformer(_atributos_de_formatacao), LogisticRegression())),

    # Experimento 1: linha de base do pré-projeto.
    Experimento("tfidf_nb", "TF-IDF (1-2) + Naive Bayes Multinomial", "1. Clássicos", "sem_stopwords",
                lambda: make_pipeline(_tfidf(), MultinomialNB()), {"multinomialnb__alpha": ALFA_NB}),
    Experimento("tfidf_svm", "TF-IDF (1-2) + SVM linear", "1. Clássicos", "sem_stopwords",
                lambda: make_pipeline(_tfidf(), LinearSVC()), {"linearsvc__C": C_SVM}),
    Experimento("tfidf_reglog", "TF-IDF (1-2) + Regressão Logística", "1. Clássicos", "sem_stopwords",
                lambda: make_pipeline(_tfidf(), LogisticRegression(max_iter=2000)),
                {"logisticregression__C": C_REGLOG}),

    # Ablações sobre a SVM: uma decisão de cada vez muda em relação a tfidf_svm.
    Experimento("svm_com_stopwords", "TF-IDF (1-2) + SVM, mantendo stopwords", "1. Clássicos: ablação",
                "com_stopwords", lambda: make_pipeline(_tfidf(), LinearSVC()), {"linearsvc__C": C_SVM}),
    Experimento("svm_radical", "TF-IDF (1-2) + SVM, com radicalização RSLP", "1. Clássicos: ablação",
                "radical", lambda: make_pipeline(_tfidf(), LinearSVC()), {"linearsvc__C": C_SVM}),
    Experimento("svm_lema", "TF-IDF (1-2) + SVM, com lematização (spaCy)", "1. Clássicos: ablação",
                "lema", lambda: make_pipeline(_tfidf(), LinearSVC()), {"linearsvc__C": C_SVM}),
    Experimento("svm_unigramas", "TF-IDF (1) + SVM, só unigramas", "1. Clássicos: ablação",
                "sem_stopwords", lambda: make_pipeline(_tfidf(ngram_range=(1, 1)), LinearSVC()),
                {"linearsvc__C": C_SVM}),
    Experimento("svm_caracteres", "TF-IDF de n-gramas de caracteres (2-5) + SVM, limpeza mínima",
                "1. Clássicos: ablação", "minimo",
                lambda: make_pipeline(_tfidf(analyzer="char_wb", ngram_range=(2, 5), min_df=3), LinearSVC()),
                {"linearsvc__C": C_SVM}),
    Experimento("svm_caracteres_neutro", "TF-IDF de n-gramas de caracteres (2-5) + SVM, artefatos neutralizados",
                "1. Clássicos: ablação", "neutro",
                lambda: make_pipeline(_tfidf(analyzer="char_wb", ngram_range=(2, 5), min_df=3), LinearSVC()),
                {"linearsvc__C": C_SVM}),
]
POR_NOME = {e.nome: e for e in EXPERIMENTOS}


def escores_fake(modelo, textos) -> np.ndarray:
    """Escore contínuo, maior = mais provável "fake" (usado na ROC-AUC)."""
    classes = list(modelo.classes_)
    if hasattr(modelo, "decision_function"):
        d = modelo.decision_function(textos)
        return d if classes[1] == config.POSITIVO else -d
    return modelo.predict_proba(textos)[:, classes.index(config.POSITIVO)]


def termos_mais_informativos(modelo, n: int = 25) -> dict[str, list[tuple[str, float]]]:
    """Termos de maior peso para cada classe num modelo linear (interpretabilidade)."""
    vetorizador, classificador = modelo[0], modelo[-1]
    nomes = vetorizador.get_feature_names_out()
    if hasattr(classificador, "coef_"):
        pesos = classificador.coef_[0] if list(classificador.classes_)[1] == config.POSITIVO else -classificador.coef_[0]
    else:  # Naive Bayes: razão de log-probabilidades entre as classes
        i = list(classificador.classes_).index(config.POSITIVO)
        pesos = classificador.feature_log_prob_[i] - classificador.feature_log_prob_[1 - i]
    ordem = np.argsort(pesos)
    return {config.POSITIVO: [(nomes[j], round(float(pesos[j]), 4)) for j in ordem[::-1][:n]],
            config.NEGATIVO: [(nomes[j], round(float(pesos[j]), 4)) for j in ordem[:n]]}


def rodar(exp: Experimento) -> None:
    print(f"\n=== {exp.nome}: {exp.descricao}")
    config.fixar_sementes()
    partes = dados.carregar_particoes(exp.versao)
    corpus_inteiro = dados.carregar(exp.versao)
    textos = dict(zip(corpus_inteiro.ids, textos_preprocessados(corpus_inteiro, exp.versao, exp.preprocessamento)))
    X = {p: [textos[i] for i in c.ids] for p, c in partes.items()}
    treino = partes["treino"]
    grupos = [d.metadados["par"] for d in treino]

    busca = GridSearchCV(exp.criar(), exp.grade or {}, scoring="accuracy", n_jobs=-1, refit=True,
                         cv=GroupKFold(n_splits=5, shuffle=True, random_state=config.SEMENTE))
    with Cronometro() as t:
        busca.fit(X["treino"], treino.rotulos, groups=grupos)
    modelo = busca.best_estimator_
    i = busca.best_index_
    with Cronometro() as t_inf:
        modelo.predict(X["teste"])

    info = {
        "familia": exp.familia, "descricao": exp.descricao, "versao_corpus": exp.versao,
        "preprocessamento": exp.preprocessamento, "melhores_hiperparametros": busca.best_params_,
        "cv_acuracia_media": round(float(busca.cv_results_["mean_test_score"][i]), 6),
        "cv_acuracia_dp": round(float(busca.cv_results_["std_test_score"][i]), 6),
        "tempo_busca_s": t.segundos, "tempo_inferencia_teste_s": t_inf.segundos,
        "modelo": str(clone(modelo)),
    }
    if isinstance(modelo[0], TfidfVectorizer):
        info["n_atributos"] = len(modelo[0].vocabulary_)
    for parte in ("validacao", "teste"):
        registrar(exp.nome, parte, partes[parte], modelo.predict(X[parte]), escores_fake(modelo, X[parte]), info)

    if isinstance(modelo[0], TfidfVectorizer):
        termos = termos_mais_informativos(modelo)
        (config.TABELAS / f"termos_{exp.nome}.json").write_text(
            json.dumps(termos, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimentos", nargs="*", help="nomes dos experimentos (padrão: todos)")
    parser.add_argument("--listar", action="store_true")
    args = parser.parse_args(argv)
    if args.listar:
        for e in EXPERIMENTOS:
            print(f"{e.nome:22} {e.familia:20} {e.descricao}")
        return
    desconhecidos = set(args.experimentos) - set(POR_NOME)
    if desconhecidos:
        parser.error(f"experimentos desconhecidos: {sorted(desconhecidos)}. Use --listar.")
    for nome in args.experimentos or POR_NOME:
        rodar(POR_NOME[nome])


if __name__ == "__main__":
    main()
