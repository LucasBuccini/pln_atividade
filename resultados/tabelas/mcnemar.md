Teste exato de McNemar no conjunto de teste; p ajustado por Holm.

| modelo_A   | modelo_B       | acc_A   | acc_B   |   A_acerta_B_erra |   A_erra_B_acerta |       p |   p_holm | significativo_5%   |
|:-----------|:---------------|:--------|:--------|------------------:|------------------:|--------:|---------:|:-------------------|
| tfidf_nb   | tfidf_svm      | 88,4    | 91,7    |                22 |                57 | 0.0001  |  0.0001  | sim                |
| tfidf_nb   | svm_caracteres | 88,4    | 94,6    |                16 |                83 | 4.4e-12 |  1.3e-11 | sim                |
| tfidf_svm  | svm_caracteres | 91,7    | 94,6    |                14 |                46 | 4.2e-05 |  8.4e-05 | sim                |
