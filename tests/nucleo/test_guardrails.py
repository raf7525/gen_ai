import pytest

from cinedata_agent.nucleo.guardrails import SQLBloqueado, validar_sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM dim_movies",
        "UPDATE dim_movies SET titulo = 'x'",
        "DROP TABLE dim_movies",
        "INSERT INTO dim_genres VALUES ('x', 'y')",
        "REPLACE INTO dim_genres VALUES ('x', 'y')",
        "CREATE TABLE t AS SELECT * FROM dim_movies",
        "ALTER TABLE dim_movies ADD COLUMN x INTEGER",
        "PRAGMA table_info(dim_movies)",
        "ATTACH DATABASE 'outro.db' AS outro",
        "VACUUM",
    ],
)
def test_bloqueia_comandos_que_nao_sao_leitura(sql):
    with pytest.raises(SQLBloqueado):
        validar_sql(sql)


def test_bloqueia_mais_de_um_comando():
    with pytest.raises(SQLBloqueado, match="único comando"):
        validar_sql("SELECT 1 FROM dim_movies; DROP TABLE dim_movies")


def test_bloqueia_tabela_fora_da_camada_gold():
    with pytest.raises(SQLBloqueado, match="Tabela desconhecida"):
        validar_sql("SELECT * FROM sqlite_master")


def test_bloqueia_funcao_perigosa():
    with pytest.raises(SQLBloqueado, match="load_extension"):
        validar_sql("SELECT load_extension('malicioso.so')")


def test_bloqueia_sql_vazio_e_invalido():
    with pytest.raises(SQLBloqueado):
        validar_sql("  ;  ")
    with pytest.raises(SQLBloqueado):
        validar_sql("SELECT FROM WHERE")


def test_permite_cte_e_aplica_limite_padrao():
    sql = validar_sql(
        "WITH top AS (SELECT sk_movie_id FROM fact_movies_performance) "
        "SELECT d.titulo FROM top JOIN dim_movies d ON d.sk_movie_id = top.sk_movie_id",
        limite_padrao=100,
    )
    assert sql.endswith("LIMIT 100")


def test_mantem_limite_pequeno_e_reduz_limite_exagerado():
    assert validar_sql("SELECT titulo FROM dim_movies LIMIT 10").endswith("LIMIT 10")
    assert validar_sql("SELECT titulo FROM dim_movies LIMIT 999999", limite_max=1000).endswith(
        "LIMIT 1000"
    )


def test_permite_union_e_ponto_e_virgula_final():
    sql = validar_sql("SELECT titulo FROM dim_movies UNION SELECT nome_genero FROM dim_genres;")
    assert "UNION" in sql
    assert sql.endswith("LIMIT 100")


def test_preserva_funcoes_de_data_do_sqlite():
    sql = validar_sql(
        "SELECT titulo FROM dim_movies "
        "WHERE ano_lancamento >= CAST(strftime('%Y', 'now') AS INTEGER) - 4"
    )
    assert "strftime('%y', 'now')" in sql.lower()


def test_devolve_o_sql_original_quando_ja_e_seguro():
    original = "SELECT titulo\nFROM dim_movies\nWHERE titulo IS NOT NULL\nLIMIT 5"
    assert validar_sql(original + ";") == original
