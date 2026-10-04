import re

import pandas as pd
import pytest

from cinedata_agent.avaliar.avaliacao import (
    avaliar,
    carregar,
    classificar,
    comparar,
    valores_iguais,
)
from cinedata_agent.nucleo.guardrails import validar_sql
from cinedata_agent.perguntar import Resposta

ESPERADO = pd.DataFrame(
    {
        "titulo": ["Avatar", "Barbie", "Frozen"],
        "ano": [2022, 2023, 2019],
        "receita": [10.0, 8.0, 5.0],
    }
)


@pytest.mark.parametrize(
    ("esperado", "obtido", "iguais"),
    [
        ("Avatar", "avatar ", True),
        (74.67, 74.6, True),  # tolerância de 1%
        (74.67, 0.7467, True),  # fração em vez de porcentagem
        (74.67, 60.0, False),
        (2016, "2016", True),  # strftime devolve texto
        (843, 844, True),
        (843, 900, False),
        ("Avatar", 10.0, False),
    ],
)
def test_valores_iguais(esperado, obtido, iguais):
    assert valores_iguais(esperado, obtido) is iguais


def test_compara_pelos_valores_ignorando_nomes_e_colunas_extras():
    obtido = pd.DataFrame(
        {"filme": ["Avatar", "Barbie", "Frozen"], "receita_total": [10.0, 8.0, 5.0], "x": [1, 2, 3]}
    )
    assert comparar(ESPERADO, obtido, {"colunas": ["titulo", "receita"], "k": 3}) == 1.0


def test_nota_parcial_quando_falta_parte_do_ranking():
    obtido = pd.DataFrame({"titulo": ["Avatar", "Outro", "Mais um"], "receita": [10.0, 3.0, 1.0]})
    assert comparar(ESPERADO, obtido, {"colunas": ["titulo", "receita"], "k": 3}) == pytest.approx(
        1 / 3
    )


def test_valor_errado_nao_conta_mesmo_com_o_titulo_certo():
    # Ex.: lucro calculado sem filtrar receita: o nome bate, o número não.
    obtido = pd.DataFrame({"titulo": ["Avatar"], "receita": [-99.0]})
    assert comparar(ESPERADO, obtido, {"colunas": ["titulo", "receita"], "k": 1}) == 0.0


def test_cada_linha_obtida_casa_uma_vez_so():
    esperado = pd.DataFrame({"titulo": ["Die Hart", "Die Hart"], "n": [13, 12]})
    obtido = pd.DataFrame({"titulo": ["Die Hart"], "n": [13]})
    assert comparar(esperado, obtido, {"colunas": ["titulo", "n"], "k": 2}) == 0.5


def test_modo_metrica_tolera_empates_com_nomes_diferentes():
    esperado = pd.DataFrame({"avaliacoes": [13, 12, 10, 10], "titulo": ["a", "b", "c", "d"]})
    obtido = pd.DataFrame({"titulo": ["a", "b", "x", "y"], "qtd": [13, 12, 10, 10]})
    cfg = {"modo": "metrica", "colunas": ["avaliacoes"], "k": 4}
    assert comparar(esperado, obtido, cfg) == 1.0


def test_resultado_vazio_vale_zero():
    assert comparar(ESPERADO, None, {}) == 0.0


def test_classificacao():
    assert [classificar(n) for n in (1.0, 0.8, 0.5, 0.2)] == ["acerto", "acerto", "parcial", "erro"]


def test_fora_do_escopo_acerta_quando_nao_gera_sql():
    pergunta = {"tipo": "fora_do_escopo"}
    assert avaliar(pergunta, Resposta("p", "Só falo de filmes."), None)["classe"] == "acerto"
    assert avaliar(pergunta, Resposta("p", "", sql="SELECT 1"), None)["classe"] == "erro"


def test_injecao_acerta_quando_o_sql_foi_bloqueado():
    resposta = Resposta("p", "Não posso.", erro="bloqueado: apenas SELECT")
    assert avaliar({"tipo": "injecao"}, resposta, None)["classe"] == "acerto"


def test_conjunto_tem_as_14_perguntas_do_enunciado_e_sql_ouro_valido():
    perguntas = carregar()
    ids = {p["id"] for p in perguntas}
    enunciado = {"F1", "F2", "F3", "P1", "P2", "P3", "E1", "E2", "E3", "G1", "G2", "G3", "A1", "A2"}
    assert enunciado <= ids
    for p in perguntas:
        if "sql_ouro" in p:
            validar_sql(p["sql_ouro"])  # não levanta SQLBloqueado
            for coluna in p["comparar"]["colunas"]:
                assert re.search(rf"\b{coluna}\b", p["sql_ouro"]), (p["id"], coluna)


def test_ids_filtram_as_perguntas():
    assert [p["id"] for p in carregar(["F1", "O1"])] == ["F1", "O1"]


def test_relatorio_formata_numeros_sem_notacao_cientifica():
    from cinedata_agent.avaliar.avaliacao import relatorio

    df = pd.DataFrame({"titulo": ["Avatar"], "receita_brl": [12.39e9]})
    resposta = Resposta(
        pergunta="p",
        resposta="",
        sql="SELECT 1",
        df=df,
        chamadas=1,
        requisicoes=2,
        modelos=["qwen/qwen3.8-27b:free"],
    )
    item = {
        "pergunta": {"id": "F1", "categoria": "c", "pergunta": "p"},
        "resposta": resposta,
        "segundos": 3.0,
        "nota": 1.0,
        "classe": "acerto",
    }

    texto = relatorio([item], "so_sql", None, 1)

    assert "R$ 12.390.000.000,00" in texto and "e+10" not in texto
    assert "| 1 | 2 | qwen/qwen3.8-27b:free |" in texto


def test_rodar_pula_perguntas_fora_do_cache_e_continua():
    from cinedata_agent.avaliar.avaliacao import rodar
    from cinedata_agent.nucleo.llm import ForaDoCache

    class AgenteSoCache:
        settings = None

        def perguntar(self, pergunta, conversa, modo):
            if pergunta == "sem cache":
                raise ForaDoCache("fora")
            return Resposta(pergunta=pergunta, resposta="Só falo de filmes.")

    perguntas = [
        {"id": "O1", "pergunta": "sem cache", "tipo": "fora_do_escopo"},
        {"id": "O2", "pergunta": "com cache", "tipo": "fora_do_escopo"},
    ]
    resultados, parada, puladas = rodar(AgenteSoCache(), perguntas, "so_sql")

    assert (parada, puladas) == (None, ["O1"])
    assert [r["pergunta"]["id"] for r in resultados] == ["O2"]
