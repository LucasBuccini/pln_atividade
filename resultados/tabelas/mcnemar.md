Teste exato de McNemar no conjunto de teste; p ajustado por Holm.

| modelo_A            | modelo_B            | acc_A   | acc_B   |   A_acerta_B_erra |   A_erra_B_acerta | p       | p_holm   | significativo_5%   |
|:--------------------|:--------------------|:--------|:--------|------------------:|------------------:|:--------|:---------|:-------------------|
| tfidf_nb            | tfidf_svm           | 88,4    | 91,7    |                22 |                57 | 0,0001  | 0,00062  | sim                |
| tfidf_nb            | svm_caracteres      | 88,4    | 94,6    |                16 |                83 | 4,4e-12 | 5,3e-11  | sim                |
| tfidf_nb            | bertimbau_congelado | 88,4    | 95,7    |                23 |               102 | 4,6e-13 | 5,9e-12  | sim                |
| tfidf_nb            | e5_congelado        | 88,4    | 91,2    |                56 |                86 | 0,015   | 0,059    | não                |
| tfidf_nb            | bertimbau_s42       | 88,4    | 99,6    |                 2 |               123 | 3,7e-34 | 8,5e-33  | sim                |
| tfidf_nb            | qwen7b_k0_calibrado | 88,4    | 75,6    |               208 |                70 | 4,6e-17 | 6,9e-16  | sim                |
| tfidf_nb            | qwen7b_k4_calibrado | 88,4    | 78,4    |               183 |                75 | 1,4e-11 | 1,4e-10  | sim                |
| tfidf_svm           | svm_caracteres      | 91,7    | 94,6    |                14 |                46 | 4,2e-05 | 0,0003   | sim                |
| tfidf_svm           | bertimbau_congelado | 91,7    | 95,7    |                26 |                70 | 8e-06   | 6,4e-05  | sim                |
| tfidf_svm           | e5_congelado        | 91,7    | 91,2    |                62 |                57 | 0,71    | 0,71     | não                |
| tfidf_svm           | bertimbau_s42       | 91,7    | 99,6    |                 2 |                88 | 6,6e-24 | 1,2e-22  | sim                |
| tfidf_svm           | qwen7b_k0_calibrado | 91,7    | 75,6    |               223 |                50 | 3e-27   | 6,3e-26  | sim                |
| tfidf_svm           | qwen7b_k4_calibrado | 91,7    | 78,4    |               196 |                53 | 1,8e-20 | 3,1e-19  | sim                |
| svm_caracteres      | bertimbau_congelado | 94,6    | 95,7    |                22 |                34 | 0,14    | 0,28     | não                |
| svm_caracteres      | e5_congelado        | 94,6    | 91,2    |                64 |                27 | 0,00013 | 0,00066  | sim                |
| svm_caracteres      | bertimbau_s42       | 94,6    | 99,6    |                 2 |                56 | 1,2e-14 | 1,7e-13  | sim                |
| svm_caracteres      | qwen7b_k0_calibrado | 94,6    | 75,6    |               236 |                31 | 3,2e-40 | 8e-39    | sim                |
| svm_caracteres      | qwen7b_k4_calibrado | 94,6    | 78,4    |               206 |                31 | 6,9e-33 | 1,5e-31  | sim                |
| bertimbau_congelado | e5_congelado        | 95,7    | 91,2    |                67 |                18 | 8,4e-08 | 7,6e-07  | sim                |
| bertimbau_congelado | bertimbau_s42       | 95,7    | 99,6    |                 1 |                43 | 5,1e-12 | 5,6e-11  | sim                |
| bertimbau_congelado | qwen7b_k0_calibrado | 95,7    | 75,6    |               241 |                24 | 3e-46   | 7,7e-45  | sim                |
| bertimbau_congelado | qwen7b_k4_calibrado | 95,7    | 78,4    |               211 |                24 | 1,6e-38 | 3,8e-37  | sim                |
| e5_congelado        | bertimbau_s42       | 91,2    | 99,6    |                 1 |                92 | 1,9e-26 | 3,8e-25  | sim                |
| e5_congelado        | qwen7b_k0_calibrado | 91,2    | 75,6    |               226 |                58 | 1,4e-24 | 2,6e-23  | sim                |
| e5_congelado        | qwen7b_k4_calibrado | 91,2    | 78,4    |               202 |                64 | 7,7e-18 | 1,2e-16  | sim                |
| bertimbau_s42       | qwen7b_k0_calibrado | 99,6    | 75,6    |               262 |                 3 | 1e-73   | 2,9e-72  | sim                |
| bertimbau_s42       | qwen7b_k4_calibrado | 99,6    | 78,4    |               232 |                 3 | 7,8e-65 | 2,1e-63  | sim                |
| qwen7b_k0_calibrado | qwen7b_k4_calibrado | 75,6    | 78,4    |                56 |                86 | 0,015   | 0,059    | não                |
