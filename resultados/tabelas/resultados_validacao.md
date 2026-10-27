Precisão, revocação e F1 referem-se à classe *falsa*. Valores em %.

**0. Viés de tamanho**

| Modelo                                                 | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:-------------------------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| TF-IDF (1-2) + SVM linear nos textos completos         | 97,6       | [96,7; 98,4] | 98,7       | 96,5        | 97,6 | 97,6       |     0,997 |
| RegLog só com o tamanho do texto (textos completos)    | 94,5       | [93,1; 95,9] | 93,6       | 95,5        | 94,6 | 94,5       |     0,983 |
| RegLog só com o tamanho do texto (textos normalizados) | 54,7       | [51,8; 57,5] | 54,8       | 54,3        | 54,5 | 54,7       |     0,559 |

**1. Clássicos**

| Modelo                                 | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:---------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| TF-IDF (1-2) + SVM linear              | 92,1       | [90,6; 93,6] | 92,4       | 91,9        | 92,1 | 92,1       |     0,979 |
| TF-IDF (1-2) + Regressão Logística     | 91,6       | [89,8; 93,2] | 92,0       | 91,1        | 91,5 | 91,6       |     0,978 |
| TF-IDF (1-2) + Naive Bayes Multinomial | 89,0       | [87,1; 90,7] | 90,4       | 87,2        | 88,8 | 89,0       |     0,956 |

**1. Clássicos: ablação**

| Modelo                                                       | Acurácia   | IC 95%       | Precisão   | Revocação   | F1   | F1 macro   |   ROC-AUC |
|:-------------------------------------------------------------|:-----------|:-------------|:-----------|:------------|:-----|:-----------|----------:|
| TF-IDF de n-gramas de caracteres (2-5) + SVM, limpeza mínima | 95,3       | [94,0; 96,5] | 96,9       | 93,5        | 95,2 | 95,3       |     0,992 |
| TF-IDF (1-2) + SVM, mantendo stopwords                       | 93,1       | [91,7; 94,6] | 94,8       | 91,3        | 93,0 | 93,1       |     0,982 |
| TF-IDF (1-2) + SVM, com radicalização RSLP                   | 92,7       | [91,1; 94,2] | 93,7       | 91,5        | 92,6 | 92,7       |     0,979 |
| TF-IDF (1-2) + SVM, com lematização (spaCy)                  | 92,3       | [90,7; 93,9] | 93,7       | 90,7        | 92,2 | 92,3       |     0,976 |
| TF-IDF (1) + SVM, só unigramas                               | 91,3       | [89,5; 93,0] | 91,9       | 90,6        | 91,2 | 91,3       |     0,974 |
