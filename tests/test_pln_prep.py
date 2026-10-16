import json
from collections import Counter

import pytest

from pln_prep import (CarregadorCSV, CarregadorDataFrame, CarregadorHuggingFace, CarregadorJSON, CarregadorPastas,
                      Corpus, Documento, EtapaTexto, NormalizarEspacos, ParaMinusculas, Pipeline, ReduzirAoRadical,
                      RemoverAcentos, RemoverEmails, RemoverHTML, RemoverNumeros, RemoverPontuacao, RemoverStopwords,
                      RemoverTokensCurtos, RemoverURLs, SubstituirPadrao, TokenizarPorRegex, pipelines_prontos)

SUJO = "<p>Veja em https://g1.globo.com/x?a=1 ou www.site.com.br: o Governo NÃO anunciou 3,5 mil medidas!</p>"


def tem_nltk():
    pytest.importorskip("nltk")


# Etapas de texto
@pytest.mark.parametrize("etapa, entrada, esperado", [
    (ParaMinusculas(), "AÇÃO Já", "ação já"),
    (RemoverURLs(), "ver https://a.com/b?c=1 e www.x.com.br fim", "ver   e   fim"),
    (RemoverURLs(), "veja https://a.com/b. Ou (www.x.com)!", "veja  . Ou ( )!"),
    (RemoverEmails(), "fale com joao.silva+pln@usp.br hoje", "fale com   hoje"),
    (RemoverNumeros(), "R$ 3.500,00 em 2024", "R$   em  "),
    (RemoverHTML(), "<b>a</b>&amp;b", " a &b"),
    (RemoverPontuacao(), "fim.início, ok!", "fim início  ok "),
    (RemoverPontuacao(manter="!-"), "guarda-chuva, ok!", "guarda-chuva  ok!"),
    (RemoverAcentos(), "ação, pêssego, Ñ", "acao, pessego, N"),
    (NormalizarEspacos(), "  a \n\t b  ", "a b"),
    (SubstituirPadrao(r"@\w+", ""), "oi @fulano tudo bem", "oi  tudo bem"),
])
def test_etapas_de_texto(etapa, entrada, esperado):
    assert etapa.aplicar(entrada) == esperado


def test_tokenizador_regex_mantem_hifen():
    assert TokenizarPorRegex().aplicar("o guarda-chuva, e o sol") == ["o", "guarda-chuva", "e", "o", "sol"]


# Validação de ordem
def test_etapa_de_tokens_sem_tokenizador_falha_na_criacao():
    with pytest.raises(ValueError, match="Tokenizador"):
        Pipeline([ParaMinusculas(), RemoverTokensCurtos()])


def test_dois_tokenizadores_falham():
    with pytest.raises(ValueError, match="tokenizado"):
        Pipeline([TokenizarPorRegex(), TokenizarPorRegex()])


def test_saida_tokens_exige_tokenizador():
    with pytest.raises(ValueError):
        Pipeline([ParaMinusculas()], saida="tokens")


def test_etapa_de_texto_depois_do_tokenizador_vale_por_token():
    p = Pipeline([TokenizarPorRegex(r"\S+"), RemoverPontuacao(), ParaMinusculas()], saida="tokens")
    assert p("Olá , MUNDO !") == ["olá", "mundo"]


# Pipeline
def test_valores_ausentes_viram_texto_vazio():
    p = Pipeline([ParaMinusculas(), TokenizarPorRegex()])
    assert p.transformar([None, float("nan"), "Oi"]) == ["", "", "oi"]


def test_etapa_criada_pelo_usuario_funciona_e_e_serializavel():
    class TrocarRT(EtapaTexto):
        def __init__(self, marcador: str = "<rt>"):
            self.marcador = marcador

        def aplicar(self, texto):
            return texto.replace("RT", self.marcador)

    p = Pipeline([TrocarRT(marcador="[RT]")])
    assert p("RT bom dia") == "[RT] bom dia"
    assert Pipeline.de_config(p.para_config())("RT") == "[RT]"


def test_salvar_e_carregar_pipeline(tmp_path):
    p = Pipeline([RemoverURLs(substituir_por=" URL "), ParaMinusculas(), TokenizarPorRegex(), RemoverTokensCurtos(3)],
                 saida="tokens")
    p.salvar(tmp_path / "pipeline.json")
    q = Pipeline.carregar(tmp_path / "pipeline.json")
    assert q.para_config() == p.para_config()
    assert q("Veja https://a.com AGORA já") == ["veja", "url", "agora"]


def test_descrever_lista_etapas_com_parametros():
    assert "RemoverPontuacao(manter='!')" in Pipeline([RemoverPontuacao(manter="!")]).descrever()


# Documento e Corpus
def corpus_exemplo(n_por_rotulo=50):
    docs = [Documento(id=f"{r}{i}", texto=f"texto {r} {i}", rotulo=r, metadados={"par": i})
            for r in ("fake", "true") for i in range(n_por_rotulo)]
    return Corpus(docs, "exemplo")


def test_documento_normaliza_id_e_texto():
    d = Documento(id=7, texto=None)
    assert d.id == "7" and d.texto == ""


def test_corpus_rejeita_ids_repetidos():
    with pytest.raises(ValueError, match="repetidos"):
        Corpus([Documento("1", "a"), Documento("1", "b")])


def test_corpus_acesso_e_analise():
    c = corpus_exemplo(3)
    assert len(c) == 6 and c[0].id == "fake0" and isinstance(c[1:3], Corpus)
    assert c.contar_rotulos() == {"fake": 3, "true": 3}
    assert "6 documentos" in c.resumo()
    assert len(c.filtrar(lambda d: d.rotulo == "true")) == 3


def test_preprocessar_mantem_id_rotulo_e_metadados_e_nao_altera_original():
    c = Corpus([Documento("1", "Olá MUNDO", "x", {"k": 1})])
    limpo = c.preprocessar(Pipeline([ParaMinusculas()]))
    assert limpo[0] == Documento("1", "olá mundo", "x", {"k": 1})
    assert c[0].texto == "Olá MUNDO"


def test_preprocessar_exige_saida_texto():
    with pytest.raises(ValueError):
        corpus_exemplo(1).preprocessar(Pipeline([TokenizarPorRegex()], saida="tokens"))


def test_pipeline_aceita_corpus_diretamente():
    assert Pipeline([ParaMinusculas()]).transformar(Corpus([Documento("1", "A")])) == ["a"]


def test_dividir_estratificado_e_reprodutivel():
    treino, val, teste = corpus_exemplo(100).dividir((0.7, 0.15, 0.15))
    assert (len(treino), len(val), len(teste)) == (140, 30, 30)
    assert treino.contar_rotulos() == {"fake": 70, "true": 70}
    assert set(treino.ids).isdisjoint(teste.ids)
    assert corpus_exemplo(100).dividir((0.7, 0.15, 0.15))[2].ids == teste.ids


def test_dividir_agrupado_mantem_pares_juntos():
    treino, teste = corpus_exemplo(100).dividir((0.8, 0.2), agrupar_por="par")
    pares_treino = {d.metadados["par"] for d in treino}
    pares_teste = {d.metadados["par"] for d in teste}
    assert pares_treino.isdisjoint(pares_teste) and len(teste) == 40
    assert all(n == 2 for n in Counter(d.metadados["par"] for d in teste).values())


def test_dividir_rejeita_proporcoes_invalidas():
    with pytest.raises(ValueError):
        corpus_exemplo(5).dividir((0.5, 0.2))


# Carregadores
def test_carregador_csv(tmp_path):
    arquivo = tmp_path / "avaliacoes.csv"
    arquivo.write_text("cod;comentario;nota;cidade\n10;Ótimo, chegou rápido;5;SP\n11;;1;RJ\n", encoding="utf-8")
    c = CarregadorCSV(arquivo, coluna_texto="comentario", coluna_rotulo="nota", coluna_id="cod",
                      separador=";").carregar()
    assert c.nome == "avaliacoes"
    assert c[0] == Documento("10", "Ótimo, chegou rápido", "5", {"cidade": "SP"})
    assert c[1].texto == ""


def test_carregador_csv_coluna_inexistente_tem_mensagem_clara(tmp_path):
    arquivo = tmp_path / "a.csv"
    arquivo.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(KeyError, match="Colunas disponíveis"):
        CarregadorCSV(arquivo, coluna_texto="texto").carregar()


def test_carregador_json_lista_e_metadados_escolhidos(tmp_path):
    arquivo = tmp_path / "noticias.json"
    arquivo.write_text(json.dumps([{"titulo": "A", "corpo": "texto a", "fonte": "g1", "ano": 2017}]), encoding="utf-8")
    c = CarregadorJSON(arquivo, coluna_texto="corpo", coluna_rotulo=None, coluna_id=None,
                       colunas_metadados=["fonte"]).carregar()
    assert c[0] == Documento("0", "texto a", None, {"fonte": "g1"})


def test_salvar_jsonl_e_recarregar_sem_argumentos(tmp_path):
    original = corpus_exemplo(2)
    original.salvar_jsonl(tmp_path / "corpus.jsonl")
    recarregado = CarregadorJSON(tmp_path / "corpus.jsonl").carregar()
    assert recarregado.documentos == original.documentos


def test_carregador_pastas(tmp_path):
    for rotulo, textos in {"fake": ["f1", "f2", "f10"], "true": ["t1", "t2", "t10"]}.items():
        (tmp_path / rotulo).mkdir()
        for t in textos:
            (tmp_path / rotulo / f"{t[1:]}.txt").write_text(f"texto {t}", encoding="utf-8")
    c = CarregadorPastas(tmp_path).carregar()
    assert c.ids[:3] == ["fake/1.txt", "fake/2.txt", "fake/10.txt"]
    assert c[0] == Documento("fake/1.txt", "texto f1", "fake", {"arquivo": "1", "pasta": "fake"})
    assert len(CarregadorPastas(tmp_path, rotulos=["true"]).carregar()) == 3


def test_carregador_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"msg": ["oi", None], "classe": ["a", "b"]})
    c = CarregadorDataFrame(df, coluna_texto="msg", coluna_rotulo="classe").carregar()
    assert c.textos == ["oi", ""] and c.rotulos == ["a", "b"]
    assert list(c.para_dataframe().columns) == ["id", "texto", "rotulo"]


def test_carregador_huggingface_converte_classlabel():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict(
        {"text": ["perdi meu cartão", "qual a taxa?"], "label": [1, 0]},
        features=datasets.Features({"text": datasets.Value("string"),
                                    "label": datasets.ClassLabel(names=["taxas", "cartao_perdido"])}))
    c = CarregadorHuggingFace(ds, coluna_texto="text", coluna_rotulo="label").carregar()
    assert c.rotulos == ["cartao_perdido", "taxas"]


# Dependências opcionais
def test_stopwords_ignoram_maiusculas_e_acentos():
    tem_nltk()
    assert RemoverStopwords().aplicar(["NÃO", "nao", "Governo", "é", "e"]) == ["Governo"]
    assert RemoverStopwords(manter=["não"]).aplicar(["nao", "governo", "o"]) == ["nao", "governo"]
    assert RemoverStopwords(adicionar=["governo"]).aplicar(["governo", "medida"]) == ["medida"]


def test_idioma_ingles():
    tem_nltk()
    assert RemoverStopwords("ingles").aplicar(["the", "card", "is", "lost"]) == ["card", "lost"]
    assert ReduzirAoRadical("ingles").aplicar(["running"]) == ["run"]


def test_pipeline_pronto_classico_com_radical():
    tem_nltk()
    saida = pipelines_prontos.para_modelos_classicos(normalizacao="radical")(SUJO)
    assert saida == "vej govern anunci mil med"


def test_pipeline_pronto_mantendo_negacao():
    tem_nltk()
    p = pipelines_prontos.para_modelos_classicos(manter_stopwords=["não"], remover_acentos=True)
    assert p("Não gostei do produto") == "nao gostei produto"


def test_pipeline_pronto_transformers_preserva_texto_natural():
    assert pipelines_prontos.para_transformers()(SUJO) == "Veja em ou : o Governo NÃO anunciou 3,5 mil medidas!"


def test_lematizar_um_lema_por_token():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("pt_core_news_sm")
    except OSError:
        pytest.skip("modelo pt_core_news_sm não instalado")
    p = pipelines_prontos.para_modelos_classicos(normalizacao="lema", saida="tokens")
    saida = p.transformar(["Os meninos estavam correndo pelas ruas", None])
    assert "menino" in saida[0] and "rua" in saida[0] and saida[1] == []


def test_dentro_do_scikit_learn_com_clone():
    tem_nltk()
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    X = ["notícia verdadeira sobre economia", "BOMBA!!! veja https://x.com", "economia cresce", "URGENTE compartilhe"]
    y = [0, 1, 0, 1]
    modelo = make_pipeline(pipelines_prontos.para_modelos_classicos(manter_stopwords=["não"]),
                           TfidfVectorizer(), MultinomialNB())
    assert list(modelo.fit(X, y).predict(X)) == y
    assert repr(clone(modelo).steps[0][1]) == repr(modelo.steps[0][1])
