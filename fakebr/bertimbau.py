"""Experimento 3: ajuste fino do BERTimbau para classificação binária.

    python -m fakebr.bertimbau                         # BERTimbau Base, sementes 42, 13 e 7
    python -m fakebr.bertimbau --sementes 42           # uma execução só (teste rápido)
    python -m fakebr.bertimbau --preprocessamento neutro --nome bertimbau_neutro   # controle de artefatos
    python -m fakebr.bertimbau --modelo neuralmind/bert-large-portuguese-cased --lote 4 --acumular 4

Protocolo (seção "Metodologia" do artigo):

- Entrada: texto com limpeza mínima (para_transformers: sem HTML/URLs/e-mails, espaços
  normalizados); caixa, acentos, pontuação e stopwords são mantidos.
- Tokenizador WordPiece do próprio modelo, truncando em 512 subpalavras (início do texto).
- Cabeça de classificação linear sobre o token [CLS], todos os pesos ajustados.
- AdamW, taxa 2e-5, decaimento linear com 10% de aquecimento, weight decay 0,01,
  4 épocas, lote efetivo 16, precisão bf16.
- Avaliação na validação ao fim de cada época; o melhor ponto (acurácia de validação)
  é restaurado no fim. O teste é usado uma única vez, depois do treino.
- Várias sementes: ajuste fino de Transformers varia entre execuções (inicialização da
  cabeça e ordem dos lotes), então o artigo reporta média e desvio-padrão.
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np

from . import config, dados
from .avaliacao import Cronometro, registrar
from .classicos import textos_preprocessados

ROTULO_ID = {config.NEGATIVO: 0, config.POSITIVO: 1}
ID_ROTULO = {v: k for k, v in ROTULO_ID.items()}


def tokenizar(tokenizador, corpus, textos: dict[str, str], max_tokens: int):
    """Dataset do HuggingFace com input_ids/attention_mask/labels e a fração de textos truncados."""
    from datasets import Dataset

    lista = [textos[i] for i in corpus.ids]
    cod = tokenizador(lista, truncation=True, max_length=max_tokens)
    completos = tokenizador(lista, truncation=False)["input_ids"]
    truncados = float(np.mean([len(c) > max_tokens for c in completos]))
    cod["labels"] = [ROTULO_ID[r] for r in corpus.rotulos]
    return Dataset.from_dict(dict(cod)), truncados


def treinar(modelo_hf: str, semente: int, args) -> None:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    nome = f"{args.nome}_s{semente}"
    print(f"\n=== {nome}: ajuste fino de {modelo_hf} (semente {semente})")
    config.fixar_sementes(semente)
    corpus = dados.carregar("normalizado")
    textos = dict(zip(corpus.ids, textos_preprocessados(corpus, "normalizado", args.preprocessamento)))
    partes = dados.dividir(corpus)

    tokenizador = AutoTokenizer.from_pretrained(modelo_hf)
    conjuntos, truncados = {}, {}
    for parte, c in partes.items():
        conjuntos[parte], truncados[parte] = tokenizar(tokenizador, c, textos, args.max_tokens)
    print("  fração truncada em", args.max_tokens, "subpalavras:", {p: round(v, 4) for p, v in truncados.items()})

    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_hf, num_labels=2, id2label=ID_ROTULO, label2id=ROTULO_ID)

    pasta = config.MODELOS / "checkpoints" / nome
    passos_por_epoca = int(np.ceil(len(conjuntos["treino"]) / (args.lote * args.acumular)))
    argumentos = TrainingArguments(
        output_dir=str(pasta),
        learning_rate=args.taxa, num_train_epochs=args.epocas, weight_decay=0.01,
        per_device_train_batch_size=args.lote, per_device_eval_batch_size=args.lote * 2,
        gradient_accumulation_steps=args.acumular,
        lr_scheduler_type="linear", warmup_steps=int(0.1 * passos_por_epoca * args.epocas),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="accuracy", greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_strategy="steps", logging_steps=50, report_to="none",
        seed=semente, data_seed=semente, dataloader_num_workers=0,
    )

    def calcular_metricas(predicao):
        logits, rotulos = predicao
        return {"accuracy": float(np.mean(np.argmax(logits, axis=-1) == rotulos))}

    treinador = Trainer(model=modelo, args=argumentos, train_dataset=conjuntos["treino"],
                        eval_dataset=conjuntos["validacao"], processing_class=tokenizador,
                        data_collator=DataCollatorWithPadding(tokenizador), compute_metrics=calcular_metricas)
    with Cronometro() as t:
        treinador.train()

    historico = [{k: v for k, v in r.items() if k in ("epoch", "loss", "eval_loss", "eval_accuracy")}
                 for r in treinador.state.log_history]
    info = {
        "familia": "3. Transformer ajustado", "descricao": args.descricao, "grupo": args.nome,
        "semente": semente, "modelo_hf": modelo_hf, "versao_corpus": "normalizado",
        "preprocessamento": args.preprocessamento, "max_tokens": args.max_tokens, "fracao_truncada": truncados,
        "hiperparametros": {"taxa": args.taxa, "epocas": args.epocas, "lote_efetivo": args.lote * args.acumular,
                            "weight_decay": 0.01, "aquecimento": 0.1, "bf16": argumentos.bf16},
        "melhor_checkpoint": treinador.state.best_model_checkpoint,
        "melhor_acuracia_validacao": treinador.state.best_metric,
        "tempo_treino_s": t.segundos, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "historico": historico,
    }
    for parte in ("validacao", "teste"):
        with Cronometro() as t_inf:
            logits = treinador.predict(conjuntos[parte]).predictions
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = prob / prob.sum(axis=1, keepdims=True)
        info[f"tempo_inferencia_{parte}_s"] = t_inf.segundos
        registrar(nome, parte, partes[parte], [ID_ROTULO[i] for i in prob.argmax(axis=1)],
                  prob[:, ROTULO_ID[config.POSITIVO]], info)

    if args.salvar_modelo:
        treinador.save_model(str(config.MODELOS / nome))
    shutil.rmtree(pasta, ignore_errors=True)   # checkpoints intermediários ocupam ~1 GB cada
    del treinador, modelo
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--nome", default="bertimbau", help="prefixo dos arquivos de resultado")
    parser.add_argument("--descricao", default="BERTimbau Base com ajuste fino")
    parser.add_argument("--sementes", type=int, nargs="+", default=[42, 13, 7])
    parser.add_argument("--taxa", type=float, default=2e-5)
    parser.add_argument("--epocas", type=int, default=4)
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--acumular", type=int, default=2, help="passos de acumulação de gradiente")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--preprocessamento", default="minimo", choices=["minimo", "neutro"],
                        help="neutro: remove os artefatos de formatação (use com --nome bertimbau_neutro)")
    parser.add_argument("--salvar-modelo", action="store_true", help="grava os pesos finais em resultados/modelos")
    args = parser.parse_args(argv)
    for semente in args.sementes:
        treinar(args.modelo, semente, args)


if __name__ == "__main__":
    main()
