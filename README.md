# Detecção de notícias falsas em português no FAKE.BR

Trabalho de PLN: comparação entre modelos estatísticos, embeddings congelados, BERTimbau com ajuste fino e LLM
(zero-shot e few-shot) no FAKE.BR-Corpus, com protocolo de avaliação único. O repositório tem duas partes:

| Pasta | O que é |
|---|---|
| [fakebr/](fakebr/) | Experimentos do artigo. Cada módulo é uma etapa e roda pela linha de comando. |
| [pln_prep/](pln_prep/) | Biblioteca genérica de pré-processamento usada pelos experimentos (documentada [abaixo](#pln_prep--pré-processamento-padronizado-para-pln)). |
| [artigo/](artigo/) | Rascunho do artigo ([artigo.md](artigo/artigo.md)) e referências ([referencias.bib](artigo/referencias.bib)). |
| [resultados/](resultados/) | Métricas (JSON), predições por notícia (CSV), tabelas (Markdown/LaTeX) e figuras (PDF/PNG). |
| [dados/particoes.json](dados/particoes.json) | Partição fixa treino/validação/teste por pares (versionada). |

## Como reproduzir

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # GPU (exp. 2-4)
pip install -r requirements-experimentos.txt && pip install -e .
python -m spacy download pt_core_news_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"

python -m fakebr.dados          # baixa o corpus (git clone) e confere as partições
python -m fakebr.exploracao     # Seção 3: estatísticas do corpus e viés de tamanho
python -m fakebr.classicos      # Exp. 0 e 1: tamanho, TF-IDF + NB/SVM/RegLog e ablações   (CPU, ~3 min)
python -m fakebr.embeddings     # Exp. 2: BERTimbau e E5 congelados + RegLog                (GPU, ~5 min)
python -m fakebr.bertimbau      # Exp. 3: ajuste fino, 3 sementes                            (GPU)
python -m fakebr.llm            # Exp. 4: Qwen2.5-7B-Instruct 4 bits, zero e few-shot        (GPU)
python -m fakebr.relatorio      # Seções 5 e 6: tabelas, McNemar, figuras e análise de erros
python -m pytest tests          # testes da biblioteca e dos experimentos
```

Cada comando aceita `--help`. Os experimentos clássicos podem ser rodados um a um
(`python -m fakebr.classicos --listar`).

## Protocolo, em resumo

- **Corpus**: versão `size_normalized_texts` (o tamanho do texto sozinho dá 94% de acurácia nos textos completos).
- **Partição**: 70/15/15 sorteada por **par** (falsa e verdadeira do mesmo assunto ficam juntas): 5.040/1.080/1.080.
- **Seleção de hiperparâmetros**: GridSearchCV com GroupKFold(5) por par no treino; a validação escolhe o ponto de
  parada do BERTimbau e o limiar do LLM; o teste é usado uma vez.
- **Avaliação** ([fakebr/avaliacao.py](fakebr/avaliacao.py)): acurácia, P/R/F1 da classe falsa, F1 macro, MCC,
  ROC-AUC; IC 95% por bootstrap reamostrando pares; McNemar exato com correção de Holm.
- **Saída**: todo experimento passa por `avaliacao.registrar`, que grava `resultados/metricas/<modelo>.json` e
  `resultados/predicoes/<modelo>_<parte>.csv` no mesmo formato.

| Módulo | Seção do artigo | Conteúdo |
|---|---|---|
| [config.py](fakebr/config.py) | – | caminhos, semente, proporções, classe positiva |
| [dados.py](fakebr/dados.py) | 3, 4.1 | download, metadados (categoria, fonte), partições por par |
| [exploracao.py](fakebr/exploracao.py) | 3 | tamanho por classe, categorias, fontes, Figura 1 |
| [classicos.py](fakebr/classicos.py) | 4.3, 5 | experimentos declarativos (pré-processamento + vetorizador + classificador + grade) |
| [embeddings.py](fakebr/embeddings.py) | 4.3, 5 | codificadores congelados + RegLog |
| [bertimbau.py](fakebr/bertimbau.py) | 4.3, 5 | ajuste fino com `transformers.Trainer` |
| [llm.py](fakebr/llm.py) | 4.3, 5 | prompt, few-shot, escore por log-probabilidade, calibração do limiar |
| [avaliacao.py](fakebr/avaliacao.py) | 4.4 | métricas, bootstrap, McNemar, registro |
| [relatorio.py](fakebr/relatorio.py) | 5, 6 | tabelas, testes pareados, figuras, erros por categoria/fonte/tamanho |
| [estilo.py](fakebr/estilo.py) | – | estilo único das figuras |

---

# pln_prep — pré-processamento padronizado para PLN

Biblioteca em Python, orientada a objetos, para carregar **qualquer dataset de texto** num formato único e aplicar
pré-processamento de forma padronizada, reprodutível e compatível com o scikit-learn.

A ideia central é separar três responsabilidades:

1. **Carregar**: cada formato de origem (CSV, JSON, pastas de `.txt`, DataFrame, HuggingFace) tem um *carregador*
   que converte os dados para o formato padrão, o `Corpus`.
2. **Pré-processar**: um `Pipeline` é uma lista de *etapas* (minúsculas, remover URLs, stopwords, lematizar...).
   Cada etapa é uma classe com uma única responsabilidade.
3. **Dividir e entregar ao modelo**: o `Corpus` sabe se dividir em treino/validação/teste e entrega `textos` (X) e
   `rotulos` (y) para qualquer modelo.

Como todo dataset passa a ter a mesma forma depois de carregado, o mesmo código de pré-processamento e de
modelagem serve para qualquer um dos temas.

---

## Sumário

1. [Instalação](#1-instalação)
2. [Fluxo completo](#2-fluxo-completo)
3. [Formato padrão de entrada: Documento e Corpus](#3-formato-padrão-de-entrada-documento-e-corpus)
4. [Carregadores](#4-carregadores)
5. [Pipeline](#5-pipeline)
6. [Etapas disponíveis](#6-etapas-disponíveis)
7. [Pipelines prontos](#7-pipelines-prontos)
8. [Ordem das etapas: regras e armadilhas](#8-ordem-das-etapas-regras-e-armadilhas)
9. [Uso com scikit-learn](#9-uso-com-scikit-learn)
10. [Como estender: etapa ou carregador novo](#10-como-estender-etapa-ou-carregador-novo)
11. [Receitas por tema do trabalho](#11-receitas-por-tema-do-trabalho)
12. [Estrutura do pacote e testes](#12-estrutura-do-pacote-e-testes)

---

## 1. Instalação

O núcleo usa só a biblioteca padrão do Python (3.10+). As dependências pesadas são opcionais e só são carregadas
quando a etapa que precisa delas é usada.

| Recurso | Precisa de |
|---|---|
| `RemoverStopwords`, `ReduzirAoRadical` | `nltk` + `nltk.download("stopwords")` e `nltk.download("rslp")` |
| `Lematizar` | `spacy` + modelo do idioma (`python -m spacy download pt_core_news_sm`) |
| `CarregadorDataFrame`, `Corpus.para_dataframe` | `pandas` |
| `CarregadorHuggingFace` | `datasets` |
| Usar o `Pipeline` dentro do scikit-learn | `scikit-learn` |

**Local** (a partir da pasta do repositório):

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -e ".[all]"           # a lib + nltk, spacy e scikit-learn
python -m spacy download pt_core_news_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"
```

**Google Colab** (depois de subir o repositório para o GitHub):

```python
!pip install "pln-prep[all] @ git+https://github.com/<usuario>/<repositorio>.git"
!python -m spacy download pt_core_news_sm
import nltk; nltk.download("stopwords"); nltk.download("rslp")
```

---

## 2. Fluxo completo

```
  ORIGEM (qualquer formato)
  CSV · JSON/JSONL · pastas de .txt · DataFrame · HuggingFace
          │
          │  Carregador*(...).carregar()
          ▼
  ┌───────────────────────────────────────────────┐
  │ Corpus  (formato padrão)                      │
  │   Documento(id, texto, rotulo, metadados)     │
  │   Documento(id, texto, rotulo, metadados)     │
  │   ...                                         │
  └───────────────────────────────────────────────┘
          │  corpus.resumo()           → conferir o que foi carregado
          │  corpus.filtrar(...)       → (opcional) remover vazios, etc.
          │
          │  corpus.preprocessar(pipeline)
          │      Pipeline = [EtapaTexto...] → Tokenizador → [EtapaTokens...]
          ▼
  Corpus pré-processado (mesmos ids, rótulos e metadados; texto limpo)
          │
          │  corpus.dividir((0.7, 0.15, 0.15), agrupar_por=...)
          ▼
  treino · validacao · teste   →  .textos (X)  e  .rotulos (y)  →  modelo
          │
          │  pipeline.salvar("pipeline.json")   → registrar o que foi feito
          ▼
  experimento reprodutível
```

Exemplo de ponta a ponta com o FAKE.BR-Corpus (testado com os dados reais):

```python
from pln_prep import CarregadorPastas, pipelines_prontos

# 1. Carregar: cada subpasta (fake/, true/) vira um rótulo
corpus = CarregadorPastas("Fake.br-Corpus/size_normalized_texts").carregar()
print(corpus.resumo())
# Corpus 'size_normalized_texts': 7200 documentos
#   rótulos: {'fake': 3600, 'true': 3600}
#   palavras por texto: média 184, mediana 157, mín 9, máx 2095
#   textos vazios: 0

# 2. Pré-processar
pipeline = pipelines_prontos.para_modelos_classicos("portugues", normalizacao="radical")
limpo = corpus.preprocessar(pipeline)
print(limpo[0].texto[:40])   # 'kát abr diz vai coloc expuls mold reclam'

# 3. Dividir mantendo cada par falsa/verdadeira (mesmo nome de arquivo) na mesma parte
treino, validacao, teste = limpo.dividir((0.7, 0.15, 0.15), agrupar_por="arquivo")
# 5040 / 1080 / 1080 documentos, metade de cada rótulo em cada parte

# 4. Modelar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
vetorizador = TfidfVectorizer(ngram_range=(1, 2))
modelo = LinearSVC().fit(vetorizador.fit_transform(treino.textos), treino.rotulos)
print(modelo.score(vetorizador.transform(teste.textos), teste.rotulos))

# 5. Registrar exatamente o pré-processamento usado
pipeline.salvar("resultados/pipeline.json")
```

---

## 3. Formato padrão de entrada: Documento e Corpus

Arquivo: [pln_prep/documento.py](pln_prep/documento.py)

### `Documento`

Um item do dataset. É imutável: o pré-processamento cria documentos novos em vez de alterar os originais.

| Campo | Tipo | Significado |
|---|---|---|
| `id` | `str` | Identificador único. Números são convertidos para string. |
| `texto` | `str` | O texto a processar. `None` e `NaN` viram `""`. |
| `rotulo` | qualquer | A resposta esperada (classe, intenção, nota...). `None` em tarefas sem rótulo, como clustering. |
| `metadados` | `dict` | Todo o resto da origem: autor, data, categoria, resumo de referência, tradução... |

```python
Documento(id="17", texto="O governo anunciou...", rotulo="fake", metadados={"categoria": "politica"})
```

### `Corpus`

Lista ordenada de `Documento` com as operações comuns a qualquer dataset. Recusa ids repetidos já na criação.
Todas as transformações devolvem um **Corpus novo**; o original não muda.

| Membro | O que faz |
|---|---|
| `Corpus(documentos, nome="")` | Cria a partir de uma lista de `Documento`. Normalmente quem cria é um carregador. |
| `len(corpus)`, `for d in corpus` | Quantidade de documentos e iteração. |
| `corpus[0]` / `corpus[10:20]` | Um `Documento` / um `Corpus` com o trecho. |
| `corpus.textos` | Lista dos textos, na ordem. É o **X** dos modelos. |
| `corpus.rotulos` | Lista dos rótulos, na ordem. É o **y** dos modelos. |
| `corpus.ids` | Lista dos ids. |
| `corpus.contar_rotulos()` | `{"fake": 3600, "true": 3600}`, do mais para o menos frequente. Mostra desbalanceamento. |
| `corpus.resumo()` | Texto com nº de documentos, rótulos, palavras por texto (média, mediana, mín, máx) e textos vazios. |
| `corpus.preprocessar(pipeline)` | Aplica o pipeline ao texto de todos os documentos, mantendo id, rótulo e metadados. Exige `saida="texto"`. |
| `corpus.filtrar(condicao)` | Mantém os documentos em que `condicao(documento)` é verdadeira. Ex.: `corpus.filtrar(lambda d: d.texto.strip())`. |
| `corpus.dividir(proporcoes, estratificar, agrupar_por, semente)` | Divide em partes. Detalhes abaixo. |
| `corpus.salvar_jsonl(caminho)` | Grava no formato padrão, uma linha JSON por documento. Lido de volta com `CarregadorJSON(caminho)`. |
| `corpus.para_dataframe()` | `pandas.DataFrame` com colunas `id`, `texto`, `rotulo` e uma por metadado. |

**`dividir`** — parâmetros:

| Parâmetro | Padrão | Significado |
|---|---|---|
| `proporcoes` | `(0.7, 0.15, 0.15)` | Fração de cada parte; precisa somar 1. Devolve uma parte por valor, ex. `(0.8, 0.2)` → treino, teste. |
| `estratificar` | `True` | Mantém a proporção de cada rótulo igual em todas as partes. |
| `agrupar_por` | `None` | Nome de um campo de `metadados`. Documentos com o mesmo valor ficam sempre na mesma parte. Evita vazamento quando há documentos relacionados: pares do FAKE.BR, várias avaliações do mesmo pedido, vários trechos do mesmo processo. |
| `semente` | `42` | A mesma semente gera sempre a mesma divisão. |

### Formato padrão em arquivo (JSONL)

`salvar_jsonl` grava, e `CarregadorJSON` lê sem nenhum argumento extra:

```json
{"id": "fake/1.txt", "texto": "Kátia Abreu diz que...", "rotulo": "fake", "metadados": {"arquivo": "1", "pasta": "fake"}}
```

Isso permite pré-processar uma vez (por exemplo, a lematização, que é lenta), salvar e reutilizar nos próximos
notebooks.

---

## 4. Carregadores

Arquivo: [pln_prep/carregadores.py](pln_prep/carregadores.py)

Todo carregador tem o mesmo uso: cria-se o objeto dizendo onde está o dado e quais colunas usar, e
`.carregar()` devolve um `Corpus`.

```
Carregador (abstrato)          → método carregar() -> Corpus
├── CarregadorTabular (abstrato) → mapeia colunas de registros (dicionários) para Documento
│   ├── CarregadorCSV
│   ├── CarregadorJSON
│   ├── CarregadorDataFrame
│   └── CarregadorHuggingFace
└── CarregadorPastas            → uma subpasta por rótulo, um arquivo por documento
```

### Parâmetros comuns dos carregadores tabulares

| Parâmetro | Padrão | Significado |
|---|---|---|
| `coluna_texto` | — (obrigatório) | Coluna com o texto. Se não existir, o erro lista as colunas disponíveis. |
| `coluna_rotulo` | `None` | Coluna com o rótulo. `None` para tarefas sem rótulo. |
| `coluna_id` | `None` | Coluna com id **único**. `None` numera 0, 1, 2... Use `None` se a coluna da origem tiver repetições. |
| `colunas_metadados` | `None` | Colunas a guardar em `metadados`. `None` guarda todas as que sobrarem; `[]` não guarda nenhuma. |
| `nome` | `""` | Nome do corpus. Se vazio, usa o nome do arquivo. |

### `CarregadorCSV(caminho, coluna_texto, ..., separador=",", codificacao="utf-8")`

Lê CSV ou TSV (`separador="\t"`). Usa o módulo `csv` da biblioteca padrão e aceita campos de texto muito longos.
Todos os valores chegam como string (inclusive o rótulo: `"5"`, não `5`).

```python
CarregadorCSV("olist_order_reviews_dataset.csv",
              coluna_texto="review_comment_message", coluna_rotulo="review_score").carregar()
```

### `CarregadorJSON(caminho, coluna_texto="texto", coluna_rotulo="rotulo", coluna_id="id", ...)`

Lê um JSON que seja uma lista de objetos, ou JSON Lines (extensão `.jsonl`, um objeto por linha). Os padrões
correspondem ao formato gravado por `Corpus.salvar_jsonl`.

```python
CarregadorJSON("acordaos.json", coluna_texto="ementa", coluna_rotulo="assunto", coluna_id="processo").carregar()
CarregadorJSON("corpus_limpo.jsonl").carregar()      # formato padrão: nenhum argumento extra
```

### `CarregadorDataFrame(dataframe, coluna_texto, ...)`

Converte um `pandas.DataFrame` já carregado. Útil quando a leitura precisa de recursos do pandas (Excel, junção
de tabelas, filtragem prévia).

```python
df = pd.read_excel("dados.xlsx")
CarregadorDataFrame(df, coluna_texto="descricao", coluna_rotulo="categoria").carregar()
```

### `CarregadorHuggingFace(dataset, coluna_texto, ..., divisao="train", config=None)`

Lê um dataset do HuggingFace Hub pelo nome, ou um objeto `datasets.Dataset` já carregado. Rótulos do tipo
`ClassLabel` (números com nome de classe) são convertidos para o nome da classe, então o Banking77 chega com
rótulos legíveis em vez de 0 a 76.

```python
CarregadorHuggingFace("csebuetnlp/xlsum", config="portuguese", divisao="train",
                      coluna_texto="text", colunas_metadados=["summary", "title"]).carregar()
```

### `CarregadorPastas(raiz, rotulos=None, padrao="*.txt", codificacao="utf-8")`

Lê arquivos de texto organizados em uma subpasta por rótulo:

```
raiz/
├── fake/   1.txt, 2.txt, ...
└── true/   1.txt, 2.txt, ...
```

- rótulo = nome da subpasta; id = caminho relativo (`"fake/1.txt"`);
- `metadados = {"arquivo": "1", "pasta": "fake"}`;
- arquivos com nome numérico são ordenados numericamente (2 antes de 10);
- `rotulos=["true"]` lê só algumas subpastas.

---

## 5. Pipeline

Arquivo: [pln_prep/pipeline.py](pln_prep/pipeline.py)

Um `Pipeline` aplica uma lista de etapas, em ordem, a cada documento.

```python
from pln_prep import Pipeline, RemoverURLs, ParaMinusculas, TokenizarPorRegex, RemoverStopwords

pipeline = Pipeline([
    RemoverURLs(),
    ParaMinusculas(),
    TokenizarPorRegex(),
    RemoverStopwords("portugues", manter=["não"]),
], saida="texto")

pipeline("Veja https://x.com: o produto NÃO chegou")   # 'veja produto não chegou'
```

| Membro | O que faz |
|---|---|
| `Pipeline(etapas, saida="texto")` | Cria e **valida a ordem** das etapas. Erros de ordem aparecem aqui, não no meio do processamento. `saida="texto"` une os tokens por espaço; `saida="tokens"` devolve listas de tokens. |
| `pipeline(texto)` | Processa um único texto. |
| `pipeline.transformar(documentos)` | Processa vários de uma vez. Aceita lista de strings, `pandas.Series` ou `Corpus`. Cada etapa processa o lote inteiro, o que acelera a lematização. |
| `pipeline.descrever()` | Lista numerada das etapas com seus parâmetros, para registrar no relatório. |
| `pipeline.salvar(caminho)` / `Pipeline.carregar(caminho)` | Grava/lê a configuração em JSON. |
| `pipeline.para_config()` / `Pipeline.de_config(config)` | O mesmo, com dicionário em vez de arquivo. |
| `fit`, `transform`, `fit_transform`, `get_params`, `set_params` | Existem só para compatibilidade com o scikit-learn ([seção 9](#9-uso-com-scikit-learn)). |

Exemplo de `descrever()`:

```
Pipeline(saida='texto')
  0. RemoverURLs(substituir_por=' ')
  1. ParaMinusculas()
  2. TokenizarPorRegex(padrao='\\w+(?:-\\w+)*')
  3. RemoverStopwords(idioma='portugues', adicionar=(), manter=['não'])
```

---

## 6. Etapas disponíveis

Arquivos: [pln_prep/base.py](pln_prep/base.py), [pln_prep/etapas_texto.py](pln_prep/etapas_texto.py),
[pln_prep/tokenizadores.py](pln_prep/tokenizadores.py), [pln_prep/etapas_tokens.py](pln_prep/etapas_tokens.py)

Toda etapa é um de três tipos, conforme o que recebe e devolve:

```
Etapa (abstrata)  → aplicar(documento), aplicar_em_lote(documentos), parametros(), para_config()
├── EtapaTexto    texto  → texto
├── Tokenizador   texto  → tokens   (lista de strings)
└── EtapaTokens   tokens → tokens
```

### Etapas de texto (`EtapaTexto`)

| Classe | Parâmetros | O que faz | Exemplo |
|---|---|---|---|
| `ParaMinusculas()` | — | Converte para minúsculas. | `"AÇÃO Já"` → `"ação já"` |
| `RemoverURLs(substituir_por=" ")` | texto que entra no lugar | Remove `http://`, `https://` e `www.`, preservando a pontuação da frase logo depois. | `"veja https://a.com/b."` → `"veja  ."` |
| `RemoverEmails(substituir_por=" ")` | idem | Remove endereços de e-mail. | `"fale com ana@usp.br"` → `"fale com  "` |
| `RemoverNumeros(substituir_por=" ")` | idem | Remove números, inclusive com `.` e `,`. | `"R$ 3.500,00 em 2024"` → `"R$   em  "` |
| `RemoverHTML()` | — | Remove tags e converte entidades HTML. | `"<b>a</b>&amp;b"` → `" a &b"` |
| `RemoverPontuacao(manter="")` | caracteres a preservar | Troca pontuação e símbolos por espaço (para não colar palavras). | `"fim.início, ok!"` → `"fim início  ok "`; com `manter="!"`: `"... ok!"` |
| `RemoverAcentos()` | — | Remove acentos e cedilha. | `"ação, pêssego"` → `"acao, pessego"` |
| `NormalizarEspacos()` | — | Junta espaços, tabs e quebras de linha repetidos; apara as pontas. | `"  a \n\t b  "` → `"a b"` |
| `SubstituirPadrao(padrao, substituir_por=" ")` | regex e substituto | Remoção genérica por expressão regular, sem precisar criar classe. | `SubstituirPadrao(r"@\w+", "")`: `"oi @fulano"` → `"oi "` |

Etapas de texto também podem vir **depois** do tokenizador: o Pipeline aplica a cada token e descarta os que
ficarem vazios.

### Tokenizadores (`Tokenizador`)

| Classe | Parâmetros | O que faz | Exemplo |
|---|---|---|---|
| `TokenizarPorRegex(padrao=r"\w+(?:-\w+)*")` | regex de um token | Extrai palavras e números; mantém palavras com hífen; descarta pontuação. | `"o guarda-chuva, e o sol"` → `["o", "guarda-chuva", "e", "o", "sol"]` |
| `TokenizarPorEspacos()` | — | Separa nos espaços; a pontuação fica grudada. | `"olá, mundo!"` → `["olá,", "mundo!"]` |

### Etapas de tokens (`EtapaTokens`)

| Classe | Parâmetros | O que faz | Exemplo |
|---|---|---|---|
| `RemoverStopwords(idioma="portugues", adicionar=(), manter=())` | idioma; palavras extras a remover; palavras da lista a preservar | Remove palavras muito frequentes (lista do NLTK). Compara ignorando maiúsculas e acentos. | `["NÃO", "Governo", "é"]` → `["Governo"]`; com `manter=["não"]` → `["NÃO", "Governo"]` |
| `RemoverTokensCurtos(tamanho_minimo=2)` | tamanho mínimo | Descarta tokens menores que o mínimo. | `["a", "ok", "x"]` → `["ok"]` |
| `ReduzirAoRadical(idioma="portugues", algoritmo=None)` | idioma; `"rslp"` ou `"snowball"` | *Stemming*: corta sufixos. RSLP para português, Snowball para os outros idiomas. Rápido; o radical nem sempre é palavra. | `["governo", "anunciou"]` → `["govern", "anunci"]` |
| `Lematizar(modelo="pt_core_news_sm", tamanho_lote=256)` | modelo spaCy; documentos por lote | Troca cada token pela forma de dicionário, usando o contexto da frase. Mais lento e mais preciso que o radical. | `["meninos", "estavam"]` → `["menino", "estar"]` |

**Idiomas**: `"portugues"` e `"ingles"` têm mapeamento completo (NLTK + modelo spaCy). Outros idiomas do NLTK
funcionam passando o nome do NLTK direto (ex.: `idioma="spanish"`); para lematizar neles, informe
`Lematizar(modelo=...)`.

**Desempenho medido** (7.200 notícias do FAKE.BR, notebook comum, CPU):

| Pipeline | Tempo |
|---|---|
| `para_transformers()` | 0,2 s |
| `para_modelos_classicos()` | 2,7 s |
| `para_modelos_classicos(normalizacao="radical")` | 7,9 s |
| `para_modelos_classicos(normalizacao="lema")` | 41 s |

O lematizador do modelo pequeno do spaCy erra em nomes próprios (`"Abreu"` → `"abrer"`, `"senadora"` →
`"senadoro"`). Se a qualidade da lematização importar, use `pt_core_news_lg` ou prefira o radical.

---

## 7. Pipelines prontos

Arquivo: [pln_prep/pipelines_prontos.py](pln_prep/pipelines_prontos.py)

### `para_modelos_classicos(idioma, normalizacao, manter_stopwords, stopwords_extras, remover_acentos, saida)`

Limpeza completa para modelos de **saco de palavras**: TF-IDF/BoW com SVM, Naive Bayes, Regressão Logística,
K-Means, LDA...

| Parâmetro | Padrão | Significado |
|---|---|---|
| `idioma` | `"portugues"` | Idioma das stopwords, do stemmer e do lematizador. |
| `normalizacao` | `None` | `None`, `"radical"` (stemming) ou `"lema"` (lematização). |
| `manter_stopwords` | `()` | Ex.: `["não", "nem", "sem"]` em análise de sentimentos. |
| `stopwords_extras` | `()` | Palavras do domínio a remover, ex.: `["reais", "produto"]`. |
| `remover_acentos` | `False` | Junta `"não"` e `"nao"`; útil em textos informais escritos sem acento. |
| `saida` | `"texto"` | `"texto"` ou `"tokens"`. |

Etapas, nesta ordem:

```
RemoverHTML → RemoverURLs → RemoverEmails → ParaMinusculas → RemoverNumeros → RemoverPontuacao
→ NormalizarEspacos → TokenizarPorRegex
→ RemoverStopwords                          (normalizacao=None)
→ RemoverStopwords → ReduzirAoRadical       (normalizacao="radical")
→ Lematizar → RemoverStopwords              (normalizacao="lema")
→ RemoverAcentos (se remover_acentos=True) → RemoverTokensCurtos(2)
```

```python
pipelines_prontos.para_modelos_classicos(normalizacao="radical")(
    "<p>Veja em https://g1.globo.com: o Governo NÃO anunciou 3,5 mil medidas!</p>")
# 'vej govern anunci mil med'
```

### `para_transformers()`

Limpeza **mínima** para BERT/BERTimbau e LLMs: `RemoverHTML → RemoverURLs → RemoverEmails → NormalizarEspacos`.
Esses modelos têm tokenizador próprio e foram pré-treinados com texto natural; maiúsculas, acentos, pontuação e
stopwords carregam informação e devem ficar.

```python
pipelines_prontos.para_transformers()(
    "<p>Veja em https://g1.globo.com: o Governo NÃO anunciou 3,5 mil medidas!</p>")
# 'Veja em : o Governo NÃO anunciou 3,5 mil medidas!'
```

---

## 8. Ordem das etapas: regras e armadilhas

O `Pipeline` **impede** na criação:

- etapa de tokens antes de um tokenizador (`ValueError: ... adicione um Tokenizador antes dela`);
- dois tokenizadores (`... o texto já foi tokenizado`);
- `saida="tokens"` sem tokenizador.

O `Pipeline` **não impede**, mas você deve cuidar:

| Regra | Por quê |
|---|---|
| Remover URLs e e-mails **antes** de remover pontuação. | Sem os `:`, `/` e `.` a regex não reconhece mais o endereço, e sobram tokens como `https`, `www`, `com`. |
| Lematizar **antes** de remover stopwords e acentos. | O modelo do spaCy infere a classe gramatical pelo contexto e foi treinado com texto acentuado. |
| Reduzir ao radical **depois** de remover stopwords. | Radicais (`"est"`) não casam com a lista de stopwords (`"estava"`). |
| Em sentimentos, `manter=["não", "nem", "sem"]`. | A lista padrão remove a negação: "não gostei" vira "gostei". |
| Em transformers, use `para_transformers()`. | Remover stopwords, pontuação ou maiúsculas tira informação que o modelo sabe usar. |
| Em NER/extração de entidades, não use limpeza agressiva. | Minúsculas, remoção de pontuação e stopwords destroem as fronteiras e posições das entidades. |
| Divida o corpus com `agrupar_por` quando houver documentos relacionados. | Evita que o modelo veja no treino um quase-duplicado do teste. |

`RemoverStopwords` compara ignorando maiúsculas e acentos, então funciona antes ou depois de `ParaMinusculas` e
`RemoverAcentos`.

---

## 9. Uso com scikit-learn

O `Pipeline` implementa `fit`/`transform`, então pode ser a primeira etapa de um pipeline do scikit-learn, e
funciona com `clone`, `cross_val_score` e `GridSearchCV`:

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

modelo = make_pipeline(
    pipelines_prontos.para_modelos_classicos(normalizacao="radical"),
    TfidfVectorizer(ngram_range=(1, 2)),
    LinearSVC(),
)
cross_val_score(modelo, treino.textos, treino.rotulos, cv=5, scoring="f1_macro")
```

Para comparar pré-processamentos no `GridSearchCV`, varie o pipeline inteiro:

```python
grade = {"pipeline": [pipelines_prontos.para_modelos_classicos(),
                      pipelines_prontos.para_modelos_classicos(normalizacao="radical")]}
```

---

## 10. Como estender: etapa ou carregador novo

### Etapa nova

1. Herde de `EtapaTexto`, `Tokenizador` ou `EtapaTokens`, conforme o que a etapa recebe e devolve.
2. Guarde cada argumento do `__init__` num atributo **de mesmo nome**.
3. Implemente `aplicar`.

Com isso, `descrever()`, `salvar()`/`carregar()` e o `clone` do scikit-learn funcionam sem mais nada.

```python
from pln_prep import EtapaTexto

class SubstituirEmojis(EtapaTexto):
    """Troca emojis frequentes por palavras: "adorei 😍" -> "adorei emoji_amor"."""

    def __init__(self, tabela: dict | None = None):
        self.tabela = tabela or {"😍": " emoji_amor ", "😡": " emoji_raiva "}

    def aplicar(self, texto: str) -> str:
        for emoji, palavra in self.tabela.items():
            texto = texto.replace(emoji, palavra)
        return texto
```

Se processar em lote for mais rápido (modelos), sobrescreva também `aplicar_em_lote(documentos)`, como faz
`Lematizar`.

Para carregar um pipeline salvo que usa uma etapa sua, a classe precisa ter sido definida (ou importada) antes
de `Pipeline.carregar`.

### Carregador novo

- Formato com registros (dicionários): herde de `CarregadorTabular` e implemente `ler_registros()`, que devolve
  os registros. O mapeamento de colunas para `Documento` já vem pronto.
- Formato diferente: herde de `Carregador` e implemente `carregar()`, devolvendo um `Corpus`.

```python
from pln_prep import CarregadorTabular

class CarregadorParquet(CarregadorTabular):
    def __init__(self, caminho, coluna_texto, coluna_rotulo=None, coluna_id=None, colunas_metadados=None, nome=""):
        super().__init__(coluna_texto, coluna_rotulo, coluna_id, colunas_metadados, nome)
        self.caminho = caminho

    def ler_registros(self):
        import pandas as pd
        return pd.read_parquet(self.caminho).to_dict("records")
```

---

## 11. Receitas por tema do trabalho

Os nomes de colunas abaixo são os mais comuns nessas bases; confira no seu arquivo (o erro de coluna
inexistente lista as disponíveis).

| Tema | Carregador | Pré-processamento | Observações |
|---|---|---|---|
| 1. Intenções (Banking77) | `CarregadorHuggingFace("PolyAI/banking77", coluna_texto="text", coluna_rotulo="label")`, ou `CarregadorCSV` com o CSV do dataset | Embeddings de sentenças: `para_transformers()`. TF-IDF + K-Means: `para_modelos_classicos("ingles", normalizacao="lema")` | Em inglês. No clustering, o rótulo serve só para avaliar (pureza, ARI), nunca como entrada. |
| 2. Fake news (FAKE.BR) | `CarregadorPastas(".../size_normalized_texts")` | Clássicos: `para_modelos_classicos(normalizacao="radical")`. BERTimbau/LLM: `para_transformers()` | `dividir(..., agrupar_por="arquivo")` mantém os pares juntos. Use a versão normalizada por causa do viés de tamanho. |
| 3. Currículos (Kaggle) | `CarregadorCSV("Resume.csv", coluna_texto="Resume_str", coluna_rotulo="Category")` | Classificação: `para_modelos_classicos("ingles")`. NER: `para_transformers()` | Para NER, não use limpeza agressiva ([seção 8](#8-ordem-das-etapas-regras-e-armadilhas)). |
| 4. Sumarização (XL-Sum) | `CarregadorHuggingFace("csebuetnlp/xlsum", config="portuguese", coluna_texto="text", colunas_metadados=["summary", "title"])` | `para_transformers()` | O resumo de referência fica em `metadados["summary"]`. Métodos extrativos por frequência podem usar `para_modelos_classicos` só para pontuar frases. |
| 5. Tradução PT-EN | `CarregadorCSV`/`CarregadorHuggingFace` com o texto de origem em `coluna_texto` e a tradução em `colunas_metadados` | `para_transformers()` ou nenhum | Não remova nada que o modelo precise traduzir. |
| 6. Jurídico (STF) | `CarregadorCSV`/`CarregadorJSON` conforme a exportação | `para_modelos_classicos(stopwords_extras=["recurso", "art", "lei"])` | Use `agrupar_por` com o número do processo se houver vários documentos por processo. |
| 7. Saúde (PubMed/SciELO) | `CarregadorJSON`/`CarregadorPastas` | NER: `para_transformers()` | Números (dosagens) são o alvo: não use `RemoverNumeros`. |
| 8. Sentimentos (Olist) | `CarregadorCSV("olist_order_reviews_dataset.csv", coluna_texto="review_comment_message", coluna_rotulo="review_score")` | `para_modelos_classicos(manter_stopwords=["não", "nem", "sem"], remover_acentos=True)` | Não use `review_id` como `coluna_id` (tem repetições). Muitos comentários são vazios: `corpus.filtrar(lambda d: d.texto.strip())`. Rótulo chega como texto (`"5"`). |
| 9. Financeiro (CVM) | `CarregadorPastas`/`CarregadorCSV` | NER: `para_transformers()`; classificação de risco: `para_modelos_classicos()` | Valores e datas são informação: avalie manter números. |

---

## 12. Estrutura do pacote e testes

```
pln_prep/
├── __init__.py             exporta tudo; `from pln_prep import ...`
├── documento.py            Documento, Corpus (formato padrão, dividir, salvar)
├── carregadores.py         Carregador, CarregadorTabular, CSV, JSON, DataFrame, HuggingFace, Pastas
├── base.py                 Etapa, EtapaTexto, Tokenizador, EtapaTokens, etapa_de_config
├── etapas_texto.py         ParaMinusculas, RemoverURLs, RemoverPontuacao, RemoverAcentos, ...
├── tokenizadores.py        TokenizarPorRegex, TokenizarPorEspacos
├── etapas_tokens.py        RemoverStopwords, RemoverTokensCurtos, ReduzirAoRadical, Lematizar, IDIOMAS
├── pipeline.py             Pipeline
└── pipelines_prontos.py    para_modelos_classicos, para_transformers
tests/
└── test_pln_prep.py        testes de todas as etapas, carregadores, Corpus e integração com scikit-learn
```

Rodar os testes:

```bash
pip install -e ".[dev]"
python -m pytest tests
```

Testes que dependem de NLTK, spaCy, pandas, datasets ou scikit-learn são pulados automaticamente se a
dependência não estiver instalada.
