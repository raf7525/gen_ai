import pytest

from cinedata_agent.nucleo.db import TABELAS
from cinedata_agent.perguntar.context import exemplos, normalizar, recuperar, semantica
from cinedata_agent.perguntar.prompts import montar_sistema


def ids(info):
    return [r["id"] for r in info.regras]


def test_normaliza_espacos_e_pontuacao_final():
    assert normalizar("  Top 10   filmes por receita??  ") == "Top 10 filmes por receita"


@pytest.mark.parametrize("termo", ["receita", "faturamento", "bilheteria"])
def test_sinonimos_de_receita_ativam_regra_de_moeda(termo):
    assert "moeda" in ids(recuperar(f"Qual o {termo} total dos filmes?"))


def test_lucro_ativa_regra_de_receita_informada():
    info = recuperar("Lucro médio por gênero")
    assert {"lucro", "generos", "titulos"} <= set(ids(info))
    assert "margem" not in ids(info)


def test_media_por_grupo_ativa_regra_de_contagem():
    assert "medias" in ids(recuperar("Nota média IMDb por ano de lançamento"))
    assert "medias" in ids(recuperar("Lucro médio por gênero"))
    assert "medias" not in ids(recuperar("Top 10 filmes com maior receita em R$"))


def test_pergunta_sobre_elenco_ativa_regra_de_pessoas():
    assert "pessoas" in ids(recuperar("Dupla ator–diretor que mais trabalhou junta"))


def test_traduz_generos_citados_e_ativa_regra_de_generos():
    info = recuperar("Quantos filmes de terror e de ficção científica?")
    assert set(info.generos) == {("Horror", "Terror"), ("Science Fiction", "Ficção científica")}
    assert "generos" in ids(info)


def test_continuacao_herda_regras_das_perguntas_anteriores():
    info = recuperar("e só os de terror?", historico=["Lucro médio por ano"])
    assert {"lucro", "tempo", "generos"} <= set(ids(info))


def test_escolhe_exemplos_parecidos():
    info = recuperar("Quais diretores fizeram mais filmes?")
    assert info.exemplos[0]["pergunta"] == "Quais diretores dirigiram mais filmes?"
    assert len(info.exemplos) <= 3


def test_prompt_traz_esquema_regras_e_exemplos_e_e_estavel():
    info = recuperar("Top 10 filmes com maior receita em R$")
    prompt = montar_sistema(info)
    assert "fact_movies_performance" in prompt
    assert "receita_brl" in prompt
    assert "Exemplos de perguntas parecidas" in prompt
    assert prompt == montar_sistema(recuperar("Top 10 filmes com maior receita em R$?"))


def test_semantica_descreve_todas_as_tabelas_e_19_generos():
    assert set(semantica()["tabelas"]) == set(TABELAS)
    assert len(semantica()["generos"]) == 19


def test_exemplos_tem_pergunta_e_sql():
    assert len(exemplos()) >= 10
    assert all(e["pergunta"] and e["sql"].lstrip().upper().startswith("SELECT") for e in exemplos())
