import sqlite3

import pytest

from cinedata_agent.nucleo.db import ErroConsulta, conectar, executar


def test_executa_select_e_devolve_dataframe(banco):
    resultado = executar("SELECT titulo FROM dim_movies ORDER BY titulo", banco)
    assert list(resultado.df["titulo"]) == ["Avatar", "Barbie", "Filme Sem Receita"]
    assert resultado.truncado is False


def test_marca_resultado_truncado_quando_passa_do_limite(banco):
    resultado = executar("SELECT titulo FROM dim_movies", banco, max_linhas=2)
    assert len(resultado.df) == 2
    assert resultado.truncado is True


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM dim_movies",
        "UPDATE dim_movies SET titulo = 'x'",
        "DROP TABLE dim_movies",
        "CREATE TABLE t (x INTEGER)",
    ],
)
def test_conexao_nao_permite_escrita(banco, sql):
    # Mesmo que um SQL de escrita passasse pelos guardrails, o banco recusa.
    with pytest.raises(ErroConsulta):
        executar(sql, banco)
    assert sqlite3.connect(banco).execute("SELECT COUNT(*) FROM dim_movies").fetchone()[0] == 3


def test_consulta_lenta_e_interrompida_pelo_timeout(banco):
    pesada = """
        WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n)
        SELECT COUNT(*) FROM n
    """
    with pytest.raises(ErroConsulta, match="interrompida"):
        executar(pesada, banco, timeout_s=0.2)


def test_erro_de_sintaxe_vira_erro_de_consulta(banco):
    with pytest.raises(ErroConsulta, match="no such column"):
        executar("SELECT coluna_que_nao_existe FROM dim_movies", banco)


def test_banco_ausente_explica_como_resolver(tmp_path):
    with pytest.raises(FileNotFoundError, match="Baixe o cinerocket.db"):
        conectar(tmp_path / "nao_existe.db")
