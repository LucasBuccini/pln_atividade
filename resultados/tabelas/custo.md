Tempo de ajuste (busca de hiperparâmetros com validação cruzada 5x, ou treino; nos embeddings inclui codificar o corpus inteiro) e de inferência nas 1.080 notícias de teste, em segundos. GPU: NVIDIA RTX 2000 Ada 16 GB. Tempos aproximados: alguns experimentos dividiram a GPU com outro.

| familia                          | modelo                                                                | ajuste_s   | inferencia_teste_s   | ms_por_noticia   | hardware         |
|:---------------------------------|:----------------------------------------------------------------------|:-----------|:---------------------|:-----------------|:-----------------|
| 0. Atalhos                       | RegLog só com o tamanho do texto (textos normalizados)                | 2,8        | 0,0                  | 0,01             | CPU (20 threads) |
| 0. Atalhos                       | RegLog só com 4 marcas de formatação (pontuação final, aspas, '. .')  | 3,0        | 0,0                  | 0,01             | CPU (20 threads) |
| 0. Atalhos                       | RegLog só com o tamanho do texto, após neutralizar o fim dos textos   | 3,2        | 0,0                  | 0,01             | CPU (20 threads) |
| 0. Atalhos                       | RegLog só com as marcas de formatação, após neutralizá-las            | 3,2        | 0,0                  | 0,01             | CPU (20 threads) |
| 1. Clássicos                     | TF-IDF (1-2) + Naive Bayes Multinomial                                | 3,4        | 0,1                  | 0,06             | CPU (20 threads) |
| 1. Clássicos                     | TF-IDF (1-2) + Regressão Logística                                    | 3,5        | 0,1                  | 0,06             | CPU (20 threads) |
| 1. Clássicos                     | TF-IDF (1-2) + SVM linear                                             | 4,3        | 0,1                  | 0,06             | CPU (20 threads) |
| 1. Clássicos: ablação            | TF-IDF (1) + SVM, só unigramas                                        | 1,1        | 0,0                  | 0,03             | CPU (20 threads) |
| 1. Clássicos: ablação            | TF-IDF (1-2) + SVM, com radicalização RSLP                            | 3,6        | 0,1                  | 0,06             | CPU (20 threads) |
| 1. Clássicos: ablação            | TF-IDF (1-2) + SVM, com lematização (spaCy)                           | 4,3        | 0,1                  | 0,07             | CPU (20 threads) |
| 1. Clássicos: ablação            | TF-IDF (1-2) + SVM, mantendo stopwords                                | 8,4        | 0,1                  | 0,10             | CPU (20 threads) |
| 1. Clássicos: ablação            | TF-IDF de n-gramas de caracteres (2-5) + SVM, limpeza mínima          | 27,2       | 0,8                  | 0,73             | CPU (20 threads) |
| 1. Clássicos: ablação            | TF-IDF de n-gramas de caracteres (2-5) + SVM, artefatos neutralizados | 36,2       | 1,1                  | 0,98             | CPU (20 threads) |
| 2. Embeddings congelados         | BERTimbau Base congelado + RegLog, artefatos neutralizados            | 72,5       | 10,5                 | 9,71             | GPU              |
| 2. Embeddings congelados         | BERTimbau Base congelado (média dos tokens) + RegLog                  | 181,4      | 26,8                 | 24,86            | GPU              |
| 2. Embeddings congelados         | Multilingual-E5 Base congelado + RegLog                               | 183,9      | 27,2                 | 25,22            | GPU              |
| 3. Transformer ajustado          | BERTimbau Base com ajuste fino, artefatos neutralizados               | 465,3      | 6,8                  | 6,29             | GPU              |
| 3. Transformer ajustado          | BERTimbau Base com ajuste fino                                        | 493,1      | 6,5                  | 5,99             | GPU              |
| 4. LLM (aprendizado no contexto) | Qwen2.5-7B-Instruct zero-shot                                         | 0,0        | 280,6                | 259,86           | GPU              |
| 4. LLM (aprendizado no contexto) | Qwen2.5-7B-Instruct few-shot (k=4)                                    | 0,0        | 1236,9               | 1145,29          | GPU              |
| 4. LLM (aprendizado no contexto) | Qwen2.5-7B-Instruct zero-shot, artefatos neutralizados                | 0,0        | 232,7                | 215,48           | GPU              |
