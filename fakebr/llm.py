"""Experimento 4: LLM de pesos abertos por aprendizado no contexto (zero-shot e few-shot).

    python -m fakebr.llm                                  # zero-shot e few-shot (k=4) com o modelo padrão
    python -m fakebr.llm --k 0                            # só zero-shot
    python -m fakebr.llm --modelo Qwen/Qwen2.5-3B-Instruct --nome qwen3b
    python -m fakebr.llm --amostra 200                    # avalia 200 pares de cada parte (teste rápido)

Protocolo (seção "Metodologia" do artigo):

- Nenhum peso é atualizado. O modelo recebe uma instrução em português e, no few-shot,
  k exemplos rotulados do treino (metade de cada classe), como turnos anteriores do diálogo.
- Resposta por verossimilhança, não por geração livre: lê-se a probabilidade do próximo
  token ser o início de "FALSA" ou de "VERDADEIRA". O escore é log P(FALSA) − log P(VERDADEIRA).
  Isso elimina respostas fora do formato e dá um escore contínuo (para a ROC-AUC).
- Duas regras de decisão, ambas registradas:
    <nome>_kK           limiar 0 (a classe mais provável), o zero-shot "puro";
    <nome>_kK_calibrado limiar escolhido na validação, corrige a tendência do modelo a
                        preferir uma das respostas (calibração contextual simples).
- Quantização em 4 bits (NF4, bitsandbytes) para caber numa GPU de 16 GB.
- Notícias truncadas em `--max-palavras` palavras; exemplos do few-shot em `--max-palavras-exemplo`.

Ressalva para o artigo: o corpus é público desde 2018 e pode ter feito parte do
pré-treinamento do modelo (contaminação), o que inflaria o resultado.
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from . import config, dados
from .avaliacao import Cronometro, registrar
from .classicos import textos_preprocessados

INSTRUCAO = (
    "Você é um checador de fatos especializado em notícias brasileiras. Sua tarefa é classificar "
    "uma notícia como FALSA (desinformação, boato, conteúdo fabricado ou distorcido) ou VERDADEIRA "
    "(jornalismo factual publicado por veículo confiável). Considere o conteúdo, o estilo e a "
    "plausibilidade. Responda com uma única palavra: FALSA ou VERDADEIRA."
)
RESPOSTA = {config.POSITIVO: "FALSA", config.NEGATIVO: "VERDADEIRA"}


def truncar(texto: str, max_palavras: int) -> str:
    palavras = texto.split()
    return texto if len(palavras) <= max_palavras else " ".join(palavras[:max_palavras]) + " [...]"


def pergunta(texto: str) -> str:
    return f"Notícia:\n\"\"\"\n{texto}\n\"\"\"\n\nA notícia é FALSA ou VERDADEIRA?"


def escolher_exemplos(treino, textos: dict[str, str], k: int, max_palavras: int, semente: int) -> list[tuple[str, str]]:
    """k exemplos do treino, metade de cada classe, de pares diferentes, em ordem alternada embaralhada."""
    if k == 0:
        return []
    sorteio = random.Random(semente)
    pares = sorteio.sample(sorted({d.metadados["par"] for d in treino}), k)
    docs = {(d.metadados["par"], d.rotulo): d for d in treino}
    # Pares distintos para fake e true: os exemplos não podem ser as duas versões do mesmo assunto.
    escolhidos = [docs[(p, config.POSITIVO if i % 2 == 0 else config.NEGATIVO)] for i, p in enumerate(pares)]
    sorteio.shuffle(escolhidos)
    return [(truncar(textos[d.id], max_palavras), RESPOSTA[d.rotulo]) for d in escolhidos]


def montar_mensagens(texto: str, exemplos: list[tuple[str, str]]) -> list[dict]:
    mensagens = [{"role": "system", "content": INSTRUCAO}]
    for ex, resp in exemplos:
        mensagens += [{"role": "user", "content": pergunta(ex)}, {"role": "assistant", "content": resp}]
    return mensagens + [{"role": "user", "content": pergunta(texto)}]


class ClassificadorLLM:
    """Carrega o LLM quantizado e devolve log P(FALSA) − log P(VERDADEIRA) para cada prompt."""

    def __init__(self, modelo_hf: str, quantizar: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizador = AutoTokenizer.from_pretrained(modelo_hf)
        self.tokenizador.padding_side = "left"   # o último token de cada prompt fica na mesma posição
        if self.tokenizador.pad_token is None:
            self.tokenizador.pad_token = self.tokenizador.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16) if quantizar else None
        self.modelo = AutoModelForCausalLM.from_pretrained(modelo_hf, quantization_config=quant,
                                                           dtype=torch.bfloat16, device_map="cuda")
        self.modelo.eval()
        # Primeiro token de cada resposta; precisam ser diferentes para a comparação fazer sentido.
        self.token = {r: self.tokenizador.encode(t, add_special_tokens=False)[0] for r, t in RESPOSTA.items()}
        if self.token[config.POSITIVO] == self.token[config.NEGATIVO]:
            raise ValueError(f"as respostas começam pelo mesmo token neste tokenizador: {self.token}")

    def prompt(self, mensagens: list[dict]) -> str:
        return self.tokenizador.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)

    def escores(self, prompts: list[str], lote: int = 4) -> np.ndarray:
        saida = []
        torch = self.torch
        # Ordena por tamanho para reduzir padding; devolve na ordem original.
        ordem = np.argsort([len(p) for p in prompts])
        for inicio in range(0, len(prompts), lote):
            idx = ordem[inicio:inicio + lote]
            entrada = self.tokenizador([prompts[i] for i in idx], return_tensors="pt", padding=True,
                                       add_special_tokens=False).to(self.modelo.device)
            with torch.inference_mode():
                logits = self.modelo(**entrada).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            esc = (logp[:, self.token[config.POSITIVO]] - logp[:, self.token[config.NEGATIVO]]).cpu().numpy()
            saida.extend(zip(idx, esc))
            if inicio // lote % 25 == 0:
                print(f"    {min(inicio + lote, len(prompts))}/{len(prompts)}", flush=True)
        return np.array([e for _, e in sorted(saida)])


def limiar_otimo(rotulos, escores) -> float:
    """Limiar que maximiza a acurácia na validação (ponto médio entre escores consecutivos)."""
    ordem = np.sort(np.unique(escores))
    candidatos = np.concatenate([[ordem[0] - 1], (ordem[:-1] + ordem[1:]) / 2, [ordem[-1] + 1]])
    real = np.asarray(rotulos) == config.POSITIVO
    acc = [np.mean((escores > c) == real) for c in candidatos]
    return float(candidatos[int(np.argmax(acc))])


def amostrar_pares(corpus, n_pares: int, semente: int):
    pares = sorted({d.metadados["par"] for d in corpus})
    escolhidos = set(random.Random(semente).sample(pares, min(n_pares, len(pares))))
    return corpus.filtrar(lambda d: d.metadados["par"] in escolhidos)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--nome", default="qwen7b", help="prefixo dos arquivos de resultado")
    parser.add_argument("--k", type=int, nargs="+", default=[0, 4], help="nº de exemplos (0 = zero-shot)")
    parser.add_argument("--max-palavras", type=int, default=400)
    parser.add_argument("--max-palavras-exemplo", type=int, default=150)
    parser.add_argument("--amostra", type=int, default=0, help="nº de pares por parte (0 = todos)")
    parser.add_argument("--lote", type=int, default=4)
    parser.add_argument("--sem-quantizar", action="store_true")
    parser.add_argument("--preprocessamento", default="minimo", choices=["minimo", "neutro"],
                        help="neutro: remove os artefatos de formatação")
    args = parser.parse_args(argv)

    config.fixar_sementes()
    corpus = dados.carregar("normalizado")
    textos = dict(zip(corpus.ids, textos_preprocessados(corpus, "normalizado", args.preprocessamento)))
    partes = dados.dividir(corpus)
    if args.amostra:
        partes = {p: (c if p == "treino" else amostrar_pares(c, args.amostra, config.SEMENTE))
                  for p, c in partes.items()}
    clf = ClassificadorLLM(args.modelo, quantizar=not args.sem_quantizar)

    for k in args.k:
        nome = f"{args.nome}_k{k}"
        print(f"\n=== {nome}: {args.modelo}, {k} exemplos")
        exemplos = escolher_exemplos(partes["treino"], textos, k, args.max_palavras_exemplo, config.SEMENTE)
        escores, tempos = {}, {}
        for parte in ("validacao", "teste"):
            prompts = [clf.prompt(montar_mensagens(truncar(textos[i], args.max_palavras), exemplos))
                       for i in partes[parte].ids]
            with Cronometro() as t:
                escores[parte] = clf.escores(prompts, args.lote)
            tempos[parte] = t.segundos
        limiar = limiar_otimo(partes["validacao"].rotulos, escores["validacao"])
        tipo = "Zero-shot" if k == 0 else f"Few-shot (k={k})"
        info = {
            "familia": "4. LLM (aprendizado no contexto)", "modelo_hf": args.modelo, "k": k,
            "quantizacao": None if args.sem_quantizar else "nf4-4bit", "versao_corpus": "normalizado",
            "preprocessamento": args.preprocessamento, "max_palavras": args.max_palavras,
            "max_palavras_exemplo": args.max_palavras_exemplo, "amostra_pares": args.amostra or "todos",
            "instrucao": INSTRUCAO, "exemplos": [(e[:80] + "...", r) for e, r in exemplos],
            "tempo_inferencia_s": tempos, "limiar_validacao": limiar,
        }
        for parte in ("validacao", "teste"):
            esc = escores[parte]
            rotular = lambda e, lim: [config.POSITIVO if x > lim else config.NEGATIVO for x in e]
            registrar(nome, parte, partes[parte], rotular(esc, 0.0), esc,
                      {**info, "descricao": f"{args.modelo.split('/')[-1]} {tipo}", "regra": "limiar 0"})
            registrar(f"{nome}_calibrado", parte, partes[parte], rotular(esc, limiar), esc,
                      {**info, "descricao": f"{args.modelo.split('/')[-1]} {tipo}, limiar calibrado",
                       "regra": "limiar da validação"})


if __name__ == "__main__":
    main()
