"""Testes do pacote fakebr: as partes em que um erro silencioso invalidaria os resultados do artigo."""
import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from fakebr import avaliacao, config
from fakebr.relatorio import holm
from pln_prep import Corpus, Documento


def doc(i, rotulo, par):
    return Documento(id=f"{rotulo}/{i}", texto=f"texto {i}", rotulo=rotulo, metadados={"par": par})


# Métricas
def test_metricas_classe_positiva_e_matriz():
    y = ["fake", "fake", "true", "true"]
    p = ["fake", "true", "true", "fake"]
    m = avaliacao.metricas(y, p, escores=[0.9, 0.4, 0.1, 0.6])
    assert m["acuracia"] == 0.5
    assert m["precisao_fake"] == 0.5 and m["revocacao_fake"] == 0.5
    assert m["matriz_confusao"] == [[1, 1], [1, 1]]   # linhas (true, fake): [[VN, FP], [FN, VP]]
    assert m["roc_auc"] == 0.75


def test_metricas_matriz_ordem_true_fake():
    m = avaliacao.metricas(["fake", "fake", "true"], ["fake", "fake", "fake"])
    assert m["matriz_confusao"] == [[0, 1], [0, 2]]


# Bootstrap
def test_bootstrap_contem_ponto_e_e_reprodutivel():
    rng = np.random.default_rng(0)
    y = np.array(["fake", "true"] * 200)
    p = np.where(rng.random(400) < 0.85, y, np.where(y == "fake", "true", "fake"))
    grupos = np.repeat(np.arange(200), 2)
    ic = avaliacao.bootstrap_ic(y, p, grupos, n_reamostras=500)
    acc = np.mean(y == p)
    assert ic["acuracia"][0] < acc < ic["acuracia"][1]
    assert ic == avaliacao.bootstrap_ic(y, p, grupos, n_reamostras=500)


def test_bootstrap_classificador_perfeito_tem_ic_degenerado():
    y = ["fake", "true"] * 10
    ic = avaliacao.bootstrap_ic(y, y, np.repeat(np.arange(10), 2), n_reamostras=100)
    assert ic["acuracia"] == [1.0, 1.0] and ic["f1_fake"] == [1.0, 1.0]


# McNemar e Holm
def test_mcnemar_conta_discordancias():
    a = [True, True, True, False, True]
    b = [True, False, False, True, True]
    r = avaliacao.mcnemar(a, b)
    assert (r["b"], r["c"]) == (2, 1)
    assert r["p_valor"] == pytest.approx(1.0)


def test_mcnemar_sem_discordancia():
    assert avaliacao.mcnemar([True, False], [True, False])["p_valor"] == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


# LLM: regra de decisão e exemplos do few-shot (não carrega modelo nenhum)
def test_limiar_otimo_separa_classes():
    from fakebr.llm import limiar_otimo
    lim = limiar_otimo(["true", "true", "fake", "fake"], np.array([-3.0, 1.0, 2.0, 5.0]))
    assert 1.0 < lim < 2.0


def test_exemplos_few_shot_balanceados_e_de_pares_distintos():
    from fakebr.llm import RESPOSTA, escolher_exemplos
    treino = Corpus([doc(i, r, i) for i in range(20) for r in ("fake", "true")])
    textos = {d.id: d.texto for d in treino}
    ex = escolher_exemplos(treino, textos, k=4, max_palavras=50, semente=1)
    respostas = [r for _, r in ex]
    assert respostas.count(RESPOSTA["fake"]) == 2 and respostas.count(RESPOSTA["true"]) == 2
    pares = [int(t.split()[1]) for t, _ in ex]
    assert len(set(pares)) == 4
    assert escolher_exemplos(treino, textos, k=0, max_palavras=50, semente=1) == []


# Partições reais (só roda se o corpus já foi baixado)
@pytest.mark.skipif(not config.CORPUS.is_dir(), reason="corpus não baixado (python -m fakebr.dados)")
def test_particoes_sem_vazamento_de_pares():
    from fakebr import dados
    partes = dados.carregar_particoes("normalizado")
    assert [len(partes[p]) for p in config.PARTES] == [5040, 1080, 1080]
    pares = {p: {d.metadados["par"] for d in c} for p, c in partes.items()}
    assert not (pares["treino"] & pares["validacao"] or pares["treino"] & pares["teste"]
                or pares["validacao"] & pares["teste"])
    for c in partes.values():
        assert c.contar_rotulos()["fake"] == c.contar_rotulos()["true"]
