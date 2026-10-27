"""Experimento 2: embeddings de sentença congelados + Regressão Logística.

    python -m fakebr.embeddings                       # todos os codificadores
    python -m fakebr.embeddings bertimbau_congelado   # só um

Degrau intermediário entre TF-IDF e ajuste fino: a representação já é contextual, mas os
pesos do Transformer não mudam; só um classificador linear é treinado sobre ela. Comparar
`bertimbau_congelado` com o BERTimbau ajustado (Experimento 3) isola o ganho do ajuste fino,
e comparar com TF-IDF isola o ganho da representação contextual.

Protocolo igual ao dos clássicos: GridSearchCV no treino com GroupKFold por par, avaliação
uma vez em validação e teste. Os vetores ficam em cache (dados/cache/emb_<nome>.npy).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, dados
from .avaliacao import Cronometro, registrar
from .classicos import assinatura_preprocessamento, escores_fake, textos_preprocessados


@dataclass
class Codificador:
    nome: str
    modelo_hf: str
    descricao: str
    prefixo: str = ""          # alguns modelos (E5) esperam um prefixo de tarefa no texto
    max_tokens: int = 512
    preprocessamento: str = "minimo"   # "neutro" remove os artefatos de formatação (ver classicos._neutro)


CODIFICADORES = [
    # Média dos vetores da última camada do BERTimbau sem ajuste fino.
    Codificador("bertimbau_congelado", "neuralmind/bert-base-portuguese-cased",
                "BERTimbau Base congelado (média dos tokens) + RegLog"),
    # Codificador multilíngue treinado especificamente para gerar embeddings de sentença.
    Codificador("e5_congelado", "intfloat/multilingual-e5-base",
                "Multilingual-E5 Base congelado + RegLog", prefixo="query: "),
    # Controle de artefatos: o mesmo BERTimbau congelado sobre o texto neutralizado.
    Codificador("bertimbau_congelado_neutro", "neuralmind/bert-base-portuguese-cased",
                "BERTimbau Base congelado + RegLog, artefatos neutralizados", preprocessamento="neutro"),
]
POR_NOME = {c.nome: c for c in CODIFICADORES}


def codificar(cod: Codificador, corpus) -> np.ndarray:
    """Vetores do corpus inteiro, na ordem de corpus.ids, com cache em disco."""
    config.garantir_pastas()
    caminho = config.CACHE / f"emb_{cod.nome}.npz"
    assinatura = assinatura_preprocessamento(cod.preprocessamento)
    if caminho.exists():
        salvo = np.load(caminho, allow_pickle=False)
        if list(salvo["ids"]) == corpus.ids and "assinatura" in salvo.files and str(salvo["assinatura"]) == assinatura:
            return salvo["vetores"]
    import torch
    from sentence_transformers import SentenceTransformer

    modelo = SentenceTransformer(cod.modelo_hf, device="cuda" if torch.cuda.is_available() else "cpu")
    modelo.max_seq_length = cod.max_tokens
    textos = [cod.prefixo + t for t in textos_preprocessados(corpus, "normalizado", cod.preprocessamento)]
    with Cronometro() as t:
        vetores = modelo.encode(textos, batch_size=32, show_progress_bar=True, normalize_embeddings=True,
                                convert_to_numpy=True)
    print(f"  {cod.nome}: {len(textos)} textos codificados em {t.segundos}s")
    np.savez(caminho, ids=np.array(corpus.ids), vetores=vetores, segundos=t.segundos, assinatura=assinatura)
    return vetores


def rodar(cod: Codificador) -> None:
    print(f"\n=== {cod.nome}: {cod.descricao}")
    config.fixar_sementes()
    corpus = dados.carregar("normalizado")
    vetores = dict(zip(corpus.ids, codificar(cod, corpus)))
    partes = dados.dividir(corpus)
    X = {p: np.stack([vetores[i] for i in c.ids]) for p, c in partes.items()}
    treino = partes["treino"]

    busca = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                         {"logisticregression__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]},
                         scoring="accuracy", n_jobs=1,  # paralelismo do joblib trava no Windows com torch carregado
                         cv=GroupKFold(n_splits=5, shuffle=True, random_state=config.SEMENTE))
    with Cronometro() as t:
        busca.fit(X["treino"], treino.rotulos, groups=[d.metadados["par"] for d in treino])
    modelo, i = busca.best_estimator_, busca.best_index_
    info = {
        "familia": "2. Embeddings congelados", "descricao": cod.descricao, "versao_corpus": "normalizado",
        "preprocessamento": cod.preprocessamento, "modelo_hf": cod.modelo_hf, "max_tokens": cod.max_tokens,
        "dimensao": int(X["treino"].shape[1]), "melhores_hiperparametros": busca.best_params_,
        "cv_acuracia_media": round(float(busca.cv_results_["mean_test_score"][i]), 6),
        "cv_acuracia_dp": round(float(busca.cv_results_["std_test_score"][i]), 6),
        "tempo_busca_s": t.segundos,
    }
    for parte in ("validacao", "teste"):
        registrar(cod.nome, parte, partes[parte], modelo.predict(X[parte]), escores_fake(modelo, X[parte]), info)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("codificadores", nargs="*", help=f"padrão: todos ({', '.join(POR_NOME)})")
    args = parser.parse_args(argv)
    if set(args.codificadores) - set(POR_NOME):
        parser.error(f"codificadores disponíveis: {list(POR_NOME)}")
    for nome in args.codificadores or POR_NOME:
        rodar(POR_NOME[nome])


if __name__ == "__main__":
    main()
