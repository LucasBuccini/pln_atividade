Acurácia no teste (%) no texto mínimo e com os artefatos de formatação neutralizados; McNemar entre as duas versões (semente 42 nos modelos com várias sementes), p ajustado por Holm.

| modelo                                          | texto_minimo   | texto_neutralizado   | diferenca_pp   | p       | p_holm   |
|:------------------------------------------------|:---------------|:---------------------|:---------------|:--------|:---------|
| RegLog só com o tamanho do texto                | 56,4           | 56,9                 | +0,5           | 0,79    | 0,79     |
| RegLog só com 4 marcas de formatação            | 83,1           | 66,7                 | -16,4          | 1,1e-17 | 6,8e-17  |
| TF-IDF de n-gramas de caracteres                | 94,6           | 93,9                 | -0,7           | 0,2     | 0,4      |
| BERTimbau Base congelado                        | 95,7           | 93,4                 | -2,3           | 0,00062 | 0,0031   |
| BERTimbau Base com ajuste fino                  | 99,5 ± 0,1     | 98,7 ± 0,3           | -0,8           | 0,013   | 0,038    |
| Qwen2.5-7B-Instruct zero-shot, limiar calibrado | 75,6           | 78,2                 | +2,6           | 0,0046  | 0,018    |
