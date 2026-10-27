"""fakebr: experimentos de detecção de notícias falsas no FAKE.BR-Corpus.

Cada módulo corresponde a uma etapa do artigo e pode ser rodado pela linha de comando:

    python -m fakebr.dados          # baixa o corpus e fixa as partições (treino/validação/teste)
    python -m fakebr.exploracao     # Seção 3: estatísticas do corpus e viés de tamanho
    python -m fakebr.classicos      # Experimentos 0 e 1: comprimento, TF-IDF + NB/SVM/RegLog, ablações
    python -m fakebr.embeddings     # Experimento 2: embeddings congelados + Regressão Logística
    python -m fakebr.bertimbau      # Experimento 3: ajuste fino do BERTimbau
    python -m fakebr.llm            # Experimento 4: LLM zero-shot e few-shot
    python -m fakebr.relatorio      # Seções 5 e 6: tabelas, figuras, testes estatísticos e erros

O pré-processamento vem da biblioteca `pln_prep`; a avaliação de todos os modelos passa por
`fakebr.avaliacao.registrar`, que garante as mesmas métricas e o mesmo formato de saída.
"""
