Precisão, revocação e F1 referem-se à classe *falsa*. Valores em %.

**0. Atalhos**

| Modelo                                                               | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:---------------------------------------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| TF-IDF (1-2) + SVM linear nos textos completos                       | 97,6       | [96,7; 98,4] | 98,7       | 96,5        | 97,6 | 97,6       |     0,997 |
| RegLog só com o tamanho do texto (textos completos)                  | 94,5       | [93,1; 95,9] | 93,6       | 95,5        | 94,6 | 94,5       |     0,983 |
| RegLog só com 4 marcas de formatação (pontuação final, aspas, '. .') | 83,8       | [81,5; 86,1] | 87,5       | 78,9        | 83,0 | 83,8       |     0,913 |
| RegLog só com as marcas de formatação, após neutralizá-las           | 69,0       | [66,5; 71,4] | 64,8       | 83,1        | 72,8 | 68,3       |     0,690 |
| RegLog só com o tamanho do texto (textos normalizados)               | 54,7       | [51,8; 57,5] | 54,8       | 54,3        | 54,5 | 54,7       |     0,559 |
| RegLog só com o tamanho do texto, após neutralizar o fim dos textos  | 54,7       | [52,1; 57,2] | 54,7       | 55,0        | 54,8 | 54,7       |     0,570 |

**1. Clássicos**

| Modelo                                 | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:---------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| TF-IDF (1-2) + SVM linear              | 92,1       | [90,6; 93,6] | 92,4       | 91,9        | 92,1 | 92,1       |     0,979 |
| TF-IDF (1-2) + Regressão Logística     | 91,6       | [89,8; 93,2] | 92,0       | 91,1        | 91,5 | 91,6       |     0,978 |
| TF-IDF (1-2) + Naive Bayes Multinomial | 89,0       | [87,1; 90,7] | 90,4       | 87,2        | 88,8 | 89,0       |     0,956 |

**1. Clássicos: ablação**

| Modelo                                                                | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:----------------------------------------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| TF-IDF de n-gramas de caracteres (2-5) + SVM, artefatos neutralizados | 95,6       | [94,3; 96,8] | 97,1       | 93,9        | 95,5 | 95,6       |     0,993 |
| TF-IDF de n-gramas de caracteres (2-5) + SVM, limpeza mínima          | 95,3       | [94,0; 96,5] | 96,9       | 93,5        | 95,2 | 95,3       |     0,992 |
| TF-IDF (1-2) + SVM, mantendo stopwords                                | 93,1       | [91,7; 94,6] | 94,8       | 91,3        | 93,0 | 93,1       |     0,982 |
| TF-IDF (1-2) + SVM, com radicalização RSLP                            | 92,7       | [91,1; 94,2] | 93,7       | 91,5        | 92,6 | 92,7       |     0,979 |
| TF-IDF (1-2) + SVM, com lematização (spaCy)                           | 92,3       | [90,7; 93,9] | 93,7       | 90,7        | 92,2 | 92,3       |     0,976 |
| TF-IDF (1) + SVM, só unigramas                                        | 91,3       | [89,5; 93,0] | 91,9       | 90,6        | 91,2 | 91,3       |     0,974 |

**2. Embeddings congelados**

| Modelo                                                     | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:-----------------------------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| BERTimbau Base congelado (média dos tokens) + RegLog       | 96,9       | [95,9; 98,0] | 97,2       | 96,7        | 96,9 | 96,9       |     0,996 |
| BERTimbau Base congelado + RegLog, artefatos neutralizados | 95,1       | [93,8; 96,3] | 95,5       | 94,6        | 95,1 | 95,1       |     0,989 |
| Multilingual-E5 Base congelado + RegLog                    | 93,0       | [91,4; 94,4] | 94,8       | 90,9        | 92,8 | 93,0       |     0,982 |

**3. Transformer ajustado**

| Modelo                                                  | Acurácia   | IC 95%     | Precisão   | Revocação   | F1         | F1 macro   | ROC-AUC       |
|:--------------------------------------------------------|:-----------|:-----------|:-----------|:------------|:-----------|:-----------|:--------------|
| BERTimbau Base com ajuste fino                          | 99,5 ± 0,1 | média ± dp | 99,8 ± 0,1 | 99,3 ± 0,1  | 99,5 ± 0,1 | 99,5 ± 0,1 | 1,000 ± 0,000 |
| BERTimbau Base com ajuste fino, artefatos neutralizados | 98,9 ± 0,2 | média ± dp | 99,6 ± 0,2 | 98,3 ± 0,4  | 98,9 ± 0,2 | 98,9 ± 0,2 | 0,999 ± 0,000 |

**4. LLM (aprendizado no contexto)**

| Modelo                                                                   | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:-------------------------------------------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| Qwen2.5-7B-Instruct few-shot (k=4), limiar calibrado                     | 78,8       | [76,5; 80,9] | 79,3       | 78,0        | 78,6 | 78,8       |     0,853 |
| Qwen2.5-7B-Instruct zero-shot, artefatos neutralizados, limiar calibrado | 78,7       | [76,3; 81,2] | 81,1       | 74,8        | 77,8 | 78,7       |     0,859 |
| Qwen2.5-7B-Instruct zero-shot, limiar calibrado                          | 76,7       | [74,1; 79,2] | 75,3       | 79,4        | 77,3 | 76,6       |     0,836 |
| Qwen2.5-7B-Instruct few-shot (k=4)                                       | 75,7       | [73,3; 77,9] | 83,6       | 64,1        | 72,5 | 75,4       |     0,853 |
| Qwen2.5-7B-Instruct zero-shot, artefatos neutralizados                   | 70,7       | [68,5; 73,0] | 90,0       | 46,7        | 61,5 | 68,9       |     0,859 |
| Qwen2.5-7B-Instruct zero-shot                                            | 69,8       | [67,6; 72,1] | 89,1       | 45,2        | 60,0 | 67,9       |     0,836 |
