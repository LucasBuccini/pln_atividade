<!--
RASCUNHO DO ARTIGO. Estrutura no padrão SBC (Resumo/Abstract, Introdução, Trabalhos
Relacionados, Corpus, Metodologia, Resultados, Análise de Erros, Discussão, Conclusão).

Convenções:
  [PREENCHER]  trecho que depende de experimento ainda não rodado;
  [CONFERIR]   afirmação a confirmar na fonte antes da submissão.
Todos os números vêm de resultados/tabelas/ (gerados por `python -m fakebr.relatorio`).
Se rodar algum experimento de novo, atualize os números a partir dessas tabelas.
-->

# Atalhos de Fonte e Estilo na Detecção de Notícias Falsas em Português: uma Comparação entre Modelos Estatísticos, BERTimbau e LLMs no FAKE.BR

**Lucas Mendes Buccini**

## Resumo

Avaliamos, sob um protocolo único, quatro famílias de classificadores para detectar notícias falsas em português no FAKE.BR-Corpus: modelos estatísticos sobre TF-IDF (Naive Bayes, SVM linear e Regressão Logística), embeddings de sentença congelados, o BERTimbau com ajuste fino e um LLM de pesos abertos usado por aprendizado no contexto. A divisão em treino, validação e teste mantém juntos os pares temáticos do corpus, e as comparações usam intervalos de confiança por bootstrap de pares e o teste de McNemar. Mostramos primeiro que, nos textos completos, um classificador que só observa o tamanho da notícia atinge 94,2% de acurácia, o que justifica a versão normalizada do corpus. Nela, a SVM sobre palavras chega a 91,7%, n-gramas de caracteres a 94,6% e embeddings congelados do BERTimbau a 95,7%; [PREENCHER: BERTimbau ajustado e LLM]. A análise de erros revela que 93% das notícias falsas vêm de um único site e que a revocação dos modelos cai de cerca de 95% nesse site para 57–64% nas demais fontes, indicando que parte do desempenho reflete o estilo editorial da fonte, e não marcas gerais de desinformação.

**Palavras-chave:** notícias falsas, classificação de textos, BERTimbau, LLM, português.

## Abstract

[PREENCHER: tradução do resumo após os números finais.]

## 1. Introdução

A circulação de notícias falsas em ambientes digitais tornou-se um problema de interesse público, com efeitos sobre processos eleitorais, campanhas de saúde e a confiança no jornalismo [shu2017; zhou2020]. No Brasil, onde redes sociais e aplicativos de mensagens estão entre as principais fontes de informação, esse conteúdo se espalha mais rápido do que as agências de checagem conseguem verificá-lo manualmente. Ferramentas de triagem automática podem reduzir esse descompasso, mas a maior parte dos recursos e resultados foi produzida para o inglês.

Este trabalho compara, no FAKE.BR-Corpus [monteiro2018], modelos de complexidade crescente, de uma linha de base estatística barata e interpretável até um modelo de linguagem de propósito geral, para medir quanto cada aumento de custo efetivamente rende. As contribuições são:

1. um protocolo de avaliação reprodutível para o FAKE.BR, com partição fixa que mantém os pares temáticos juntos, seleção de hiperparâmetros por validação cruzada agrupada, intervalos de confiança por bootstrap de pares e testes pareados com correção para comparações múltiplas;
2. a comparação de quatro famílias de modelos sob esse protocolo: TF-IDF com classificadores lineares, embeddings congelados, BERTimbau ajustado e LLM em zero-shot e few-shot;
3. a quantificação de dois atalhos (*shortcuts*) [geirhos2020] presentes no corpus: o tamanho do texto, já conhecido, e a concentração das notícias falsas numa única fonte, cujo efeito sobre a generalização medimos.

O código, as partições e as predições de todos os modelos estão disponíveis no repositório do trabalho.

## 2. Trabalhos Relacionados

**FAKE.BR.** Monteiro et al. [monteiro2018] apresentaram o corpus e obtiveram cerca de 89% de F1 com SVM sobre atributos linguísticos e saco de palavras na versão normalizada [CONFERIR valor exato e métrica]. Silva et al. [silva2020] ampliaram o estudo com diversas representações e classificadores [CONFERIR melhores resultados].

**Classificação de textos clássica.** TF-IDF [salton1988] com SVM linear [joachims1998] ou Naive Bayes Multinomial [mccallum1998] é uma linha de base forte em espaços esparsos de alta dimensionalidade.

**Transformers para o português.** O BERTimbau [souza2020] é um BERT [devlin2019] pré-treinado no corpus brWaC e é o modelo de referência para tarefas de classificação em português brasileiro. Codificadores multilíngues treinados para gerar embeddings de sentença, como o Multilingual-E5 [wang2024e5], permitem usar representações contextuais sem ajuste fino.

**LLMs por aprendizado no contexto.** Modelos de linguagem grandes resolvem tarefas a partir de instruções e poucos exemplos [brown2020], mas são sensíveis à forma do prompt e tendem a preferir certas respostas; a calibração contextual corrige parte desse viés [zhao2021]. A possível presença dos dados de avaliação no pré-treinamento (contaminação) é uma ressalva conhecida [sainz2023].

## 3. Corpus

O FAKE.BR-Corpus [monteiro2018] reúne 7.200 notícias em português brasileiro publicadas entre 2016 e 2018: 3.600 falsas e 3.600 verdadeiras. Cada notícia falsa foi manualmente alinhada a uma verdadeira sobre o mesmo assunto, formando 3.600 pares. A Tabela 1 resume o corpus.

**Tabela 1.** Palavras por notícia, por classe e versão do corpus (`resultados/tabelas/corpus_tamanho.md`).

| Versão | Classe | n | Média | Mediana | Desvio |
|---|---|---|---|---|---|
| Completa | Falsa | 3.600 | 185,9 | 157 | 127,6 |
| Completa | Verdadeira | 3.600 | 1.100,3 | 918 | 683,0 |
| Normalizada | Falsa | 3.600 | 182,8 | 157 | 115,7 |
| Normalizada | Verdadeira | 3.600 | 184,3 | 158 | 116,1 |

**Viés de tamanho.** Nos textos completos, a notícia verdadeira é mais longa que a falsa em 98,2% dos pares, e a razão mediana entre os tamanhos é 6,1 (Figura 1). Os autores disponibilizam uma versão normalizada, na qual o texto mais longo de cada par é truncado ao número de palavras do mais curto; nela, a razão mediana cai para 1,01. Todos os experimentos principais usam a versão normalizada.

**Figura 1.** Distribuição do número de palavras por classe nas duas versões (`resultados/figuras/distribuicao_tamanho.pdf`).

**Categorias e fontes.** As notícias se distribuem em seis categorias: política (58%), TV e celebridades (21%), sociedade e cotidiano (18%), ciência e tecnologia, economia e religião (`corpus_categorias.md`). As fontes, porém, são muito desiguais entre as classes: as notícias falsas vêm de apenas 6 sites, e **3.335 das 3.600 (92,6%) de um único domínio** (diariodobrasil.org), enquanto as verdadeiras vêm de 25 domínios, com predominância de G1 e Estadão (`corpus_fontes.md`). Essa concentração permite que um classificador aprenda o estilo editorial de um site em vez de marcas gerais de desinformação, o que examinamos na Seção 6.

**Qualidade dos dados.** As duas versões não têm exatamente a mesma numeração de pares: a versão completa não tem os pares 697 e 1468, a normalizada não tem os pares 586 e 1607, e ambas vão até 3602. Verificamos que o número N corresponde à mesma notícia nas duas versões (diferenças apenas de aspas e símbolos), de modo que os metadados da versão completa valem para a normalizada, exceto em dois pares sem metadados.

## 4. Metodologia

### 4.1 Particionamento

Os 3.600 pares da versão normalizada foram sorteados uma única vez (semente 42) em treino, validação e teste na proporção 70/15/15, resultando em 5.040, 1.080 e 1.080 notícias, metade de cada classe em cada parte. A unidade sorteada é o par: as duas notícias de um mesmo assunto ficam sempre na mesma parte, para que o tema de uma notícia de teste nunca seja visto no treino por meio da sua contraparte. A partição está gravada em `dados/particoes.json` e é a mesma para todos os modelos.

O treino é usado para ajustar os modelos e escolher hiperparâmetros; a validação, para escolher o ponto de parada do BERTimbau e o limiar de decisão do LLM; o teste é usado uma única vez, ao final.

### 4.2 Pré-processamento

O pré-processamento foi implementado como pipelines declarativos e serializáveis (biblioteca `pln_prep`, desenvolvida para este trabalho), em duas configurações:

- **Completo** (modelos de saco de palavras): remoção de HTML, URLs e e-mails; conversão para minúsculas; remoção de números e pontuação; tokenização por expressão regular; remoção de stopwords (lista do NLTK); descarte de tokens com menos de 2 caracteres. Variantes com radicalização RSLP [orengo2001] e com lematização (spaCy, `pt_core_news_sm`) foram avaliadas como ablação.
- **Mínimo** (Transformers, LLM e n-gramas de caracteres): apenas remoção de HTML, URLs e e-mails e normalização de espaços. Caixa, acentos, pontuação e stopwords são mantidos, porque esses modelos têm tokenizador próprio e foram pré-treinados com texto natural.

### 4.3 Modelos

**Experimento 0: viés de tamanho.** Regressão Logística com dois atributos, log do número de palavras e log do número de caracteres, treinada nas duas versões do corpus. Mede quanto do problema se resolve sem ler o conteúdo.

**Experimento 1: modelos estatísticos.** TF-IDF de unigramas e bigramas (`min_df`=2, `max_df`=0,95, tf sublinear) com Naive Bayes Multinomial, SVM linear e Regressão Logística. Ablações sobre a SVM, mudando uma decisão por vez: manter stopwords; radicalização; lematização; só unigramas; e TF-IDF de n-gramas de caracteres (2 a 5, dentro dos limites de palavra) com pré-processamento mínimo. O hiperparâmetro de regularização (C ou α) é escolhido por busca em grade com validação cruzada de 5 dobras no treino, agrupada por par (GroupKFold), de modo que as notícias de um par nunca se dividam entre treino e validação de uma dobra. O TF-IDF é ajustado dentro de cada dobra.

**Experimento 2: embeddings congelados.** Vetores de sentença do BERTimbau Base sem ajuste fino (média dos vetores da última camada) e do Multilingual-E5 Base [wang2024e5], ambos com até 512 subpalavras, seguidos de padronização e Regressão Logística (C escolhido como no Experimento 1). Esse degrau separa o ganho da representação contextual (comparação com o TF-IDF) do ganho do ajuste fino (comparação com o Experimento 3).

**Experimento 3: BERTimbau com ajuste fino.** BERTimbau Base (`neuralmind/bert-base-portuguese-cased`, 110 M de parâmetros) com uma camada de classificação sobre o token [CLS] e todos os pesos ajustados. AdamW com taxa de aprendizado 2×10⁻⁵, decaimento linear com 10% de aquecimento, weight decay 0,01, lote efetivo 16, 4 épocas, precisão bf16 e entradas truncadas em 512 subpalavras. Ao fim de cada época o modelo é avaliado na validação, e o melhor ponto é restaurado. Como o ajuste fino varia entre execuções, treinamos com três sementes (42, 13 e 7) e reportamos média e desvio-padrão. [PREENCHER: fração de textos truncados, de `resultados/metricas/bertimbau_s42.json`, campo `fracao_truncada`.]

**Experimento 4: LLM por aprendizado no contexto.** Qwen2.5-7B-Instruct [qwen2025], quantizado em 4 bits (NF4) [dettmers2023] para caber numa GPU de 16 GB, sem nenhuma atualização de pesos. O prompt de sistema descreve a tarefa em português e pede uma única palavra, FALSA ou VERDADEIRA; no few-shot, quatro exemplos do treino (dois de cada classe, de pares distintos) entram como turnos anteriores do diálogo. Em vez de gerar texto e interpretar a resposta, lemos a distribuição do próximo token e usamos como escore a diferença log P(FALSA) − log P(VERDADEIRA). Isso evita respostas fora do formato e fornece um escore contínuo. Reportamos duas regras de decisão: limiar zero (a resposta mais provável) e limiar escolhido na validação, uma forma simples de calibração contextual [zhao2021]. As notícias são truncadas em 400 palavras e os exemplos do few-shot em 150.

### 4.4 Avaliação

Como as classes são balanceadas, a métrica principal é a acurácia. Reportamos também precisão, revocação e F1 da classe falsa, F1 macro, coeficiente de Matthews e ROC-AUC. A distinção entre erros importa para a aplicação: um falso negativo deixa uma notícia falsa circular, e um falso positivo rotula como falso um conteúdo legítimo.

Os intervalos de confiança de 95% são obtidos por bootstrap percentil [efron1993] com 2.000 reamostras, **reamostrando pares** e não notícias isoladas, já que as duas notícias de um par não são independentes. Para comparar modelos nas mesmas notícias de teste usamos o teste exato de McNemar [dietterich1998], com correção de Holm [holm1979] para as comparações múltiplas.

Todos os experimentos rodaram numa NVIDIA RTX 2000 Ada (16 GB), com as versões de bibliotecas listadas em `requirements-experimentos.txt`.

## 5. Resultados

A Tabela 2 apresenta o desempenho no teste (versão completa e em LaTeX em `resultados/tabelas/resultados_teste.md` e `.tex`), e a Figura 2 resume as acurácias com seus intervalos de confiança.

**Tabela 2.** Acurácia e F1 da classe falsa no teste (%), com IC 95% por bootstrap de pares.

| Família | Modelo | Acurácia | IC 95% | F1 (falsa) | ROC-AUC |
|---|---|---|---|---|---|
| 0. Viés de tamanho | RegLog só com tamanho, textos completos | 94,2 | [92,7; 95,6] | 94,2 | 0,981 |
| | RegLog só com tamanho, textos normalizados | 56,4 | [53,3; 59,2] | 55,4 | 0,598 |
| | TF-IDF + SVM, textos completos | 97,1 | [96,0; 98,1] | 97,1 | 0,997 |
| 1. Clássicos | TF-IDF + Naive Bayes | 88,4 | [86,7; 90,2] | 88,2 | 0,950 |
| | TF-IDF + Regressão Logística | 91,4 | [89,8; 93,0] | 91,4 | 0,973 |
| | TF-IDF + SVM linear | 91,7 | [90,0; 93,2] | 91,7 | 0,972 |
| | n-gramas de caracteres + SVM | 94,6 | [93,2; 95,9] | 94,5 | 0,987 |
| 2. Embeddings congelados | BERTimbau congelado + RegLog | 95,7 | [94,4; 96,9] | 95,7 | 0,993 |
| | Multilingual-E5 congelado + RegLog | [PREENCHER] | | | |
| 3. Transformer ajustado | BERTimbau (média ± dp, 3 sementes) | [PREENCHER] | | | |
| 4. LLM | Qwen2.5-7B zero-shot (limiar calibrado) | [PREENCHER] | | | |
| | Qwen2.5-7B few-shot k=4 (limiar calibrado) | [PREENCHER] | | | |

**Figura 2.** Acurácia no teste por modelo (`resultados/figuras/comparacao_acuracia.pdf`).

**O viés de tamanho é real e forte.** Só com o tamanho do texto, uma Regressão Logística acerta 94,2% das notícias completas, e a SVM sobre TF-IDF chega a 97,1%, valor que superestimaria o problema. Na versão normalizada, o tamanho sozinho cai para 56,4%, ainda um pouco acima do acaso (os textos foram igualados em palavras, não em caracteres), e os resultados a seguir passam a refletir o conteúdo.

**Linha de base estatística.** A SVM linear (91,7%) e a Regressão Logística (91,4%) são equivalentes, e ambas superam o Naive Bayes (88,4%; McNemar com p ajustado < 0,001). O resultado é compatível com o relatado por Monteiro et al. [monteiro2018] [CONFERIR].

**Ablações.** Nenhuma normalização morfológica ajudou: a radicalização (90,1%) e a lematização (89,7%) ficaram abaixo da SVM sem normalização, e manter as stopwords não fez diferença (91,9%). Bigramas acrescentaram 0,7 ponto em relação a só unigramas. A maior mudança veio de **n-gramas de caracteres com pré-processamento mínimo (94,6%)**, significativamente melhor que a SVM sobre palavras (p ajustado < 0,001): pontuação, caixa e grafia carregam sinal que a limpeza completa descarta.

**Representações contextuais.** Mesmo sem ajuste fino, os embeddings do BERTimbau com uma Regressão Logística (95,7%) superam todos os modelos estatísticos. [PREENCHER: E5; ganho do ajuste fino; comparação BERTimbau × LLM; custo de treino e inferência de cada família.]

**Interpretabilidade.** Entre os termos de maior peso da SVM (`termos_tfidf_svm.md`), os associados às notícias verdadeiras são majoritariamente convenções do texto jornalístico profissional, como referências de tempo ("nesta terça", "quarta-feira") e atribuição ("segundo"). Do lado das falsas aparecem nomes de políticos ("lula", "dilma") e marcas de estilo como "pra" e "coréia", grafia anterior ao Acordo Ortográfico, enquanto as verdadeiras usam "coreia". São indícios de que o modelo captura o estilo das fontes, e não a veracidade das afirmações.

## 6. Análise de Erros

**Por fonte.** Medimos a revocação da classe falsa separadamente para as notícias falsas do site dominante e para as das demais fontes (`erros_por_fonte.md`):

| Notícias falsas de… | n (teste) | NB | SVM | Caracteres + SVM | BERTimbau congelado | [PREENCHER] |
|---|---|---|---|---|---|---|
| diariodobrasil.org | 496 | 88,9 | 94,2 | 96,0 | [ver tabela] | |
| outras 5 fontes | 44 | 56,8 | 63,6 | 59,1 | [ver tabela] | |

A queda de cerca de 30 pontos indica que boa parte do desempenho vem de reconhecer o estilo editorial de um único site. O conjunto de outras fontes é pequeno (44 notícias), mas a diferença é grande e consistente entre os modelos. [PREENCHER: os modelos pré-treinados reduzem essa diferença?]

**Por categoria e tamanho.** [PREENCHER a partir de `erros_por_categoria.md` e `erros_por_tamanho.md`. Economia e religião têm poucas notícias de teste (12 e 6), então diferenças ali não são conclusivas.]

**Erros comuns.** [PREENCHER: número de notícias erradas por todos os modelos principais (`erros_comuns.csv`) e análise qualitativa de uma amostra: sátira, notícias falsas que copiam o estilo jornalístico, notícias verdadeiras de tom opinativo.]

## 7. Discussão e Limitações

- **Validade externa.** As notícias falsas vêm de poucos sites e de um período curto (2016–2018). Os resultados medem o desempenho dentro do corpus e não garantem funcionamento em notícias atuais ou de outras fontes; a Seção 6 mostra a queda fora do site dominante.
- **Normalização de tamanho.** Truncar o texto mais longo de cada par remove o atalho de tamanho, mas também corta o final das notícias verdadeiras, o que pode criar outras diferenças de estrutura (por exemplo, frases interrompidas).
- **Contaminação do LLM.** O corpus é público desde 2018 e pode ter feito parte do pré-treinamento do Qwen2.5, o que inflaria seu resultado.
- **Rótulo binário.** Sátiras e textos parcialmente verdadeiros não se acomodam bem em duas classes.
- **Truncamento.** O BERTimbau lê no máximo 512 subpalavras. [PREENCHER: fração truncada.]

## 8. Conclusão

[PREENCHER após os Experimentos 3 e 4: qual família oferece o melhor equilíbrio entre desempenho, custo e interpretabilidade; recomendação prática; trabalhos futuros, como avaliação fora da fonte (*leave-source-out*), um corpus mais recente e explicações por atribuição de relevância nos Transformers.]

## Referências

Ver `artigo/referencias.bib` (chaves entre colchetes no texto).
