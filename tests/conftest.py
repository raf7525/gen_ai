"""Fixtures comuns. Nenhum teste acessa a rede: a cota do OpenRouter é preciosa."""

import sqlite3

import httpx
import pytest

ESQUEMA = """
CREATE TABLE dim_movies (sk_movie_id TEXT PRIMARY KEY, id_filme TEXT, titulo TEXT,
  data_lancamento DATE, ano_lancamento INTEGER, duracao_minutos INTEGER,
  idioma_original TEXT, status_filme TEXT, sinopse TEXT, url_poster TEXT, url_backdrop TEXT);
CREATE TABLE fact_movies_performance (sk_movie_id TEXT PRIMARY KEY, orcamento_usd NUMERIC,
  receita_usd NUMERIC, lucro_usd NUMERIC, orcamento_brl NUMERIC, receita_brl NUMERIC,
  lucro_brl NUMERIC, popularidade REAL, nota_tmdb REAL, qtd_tmdb INTEGER, nota_imdb REAL,
  qtd_imdb INTEGER);
CREATE TABLE dim_genres (sk_genre_id TEXT PRIMARY KEY, nome_genero TEXT);
CREATE TABLE dim_people (sk_person_id TEXT PRIMARY KEY, nome_pessoa TEXT, tipo_pessoa TEXT);
CREATE TABLE dim_companies (sk_company_id TEXT PRIMARY KEY, nome_produtora TEXT);
CREATE TABLE dim_reviews (sk_review_id TEXT PRIMARY KEY, sk_movie_id TEXT,
  qtd_avaliacoes_usuarios INTEGER, nota_media_usuarios REAL);
CREATE TABLE movie_reviews (id INTEGER PRIMARY KEY, sk_movie_review_id TEXT, sk_movie_id TEXT,
  name TEXT, rating REAL, text TEXT, created_at TEXT);
CREATE TABLE bridge_movie_genre (sk_movie_id TEXT, sk_genre_id TEXT);
CREATE TABLE bridge_movie_person (sk_movie_id TEXT, sk_person_id TEXT);
CREATE TABLE bridge_movie_company (sk_movie_id TEXT, sk_company_id TEXT);

INSERT INTO dim_movies (sk_movie_id, id_filme, titulo, ano_lancamento, status_filme) VALUES
  ('m1', '1', 'Avatar', 2022, 'Lançado'),
  ('m2', '2', 'Barbie', 2023, 'Lançado'),
  ('m3', '3', 'Filme Sem Receita', 2021, 'Lançado');
INSERT INTO fact_movies_performance VALUES
  ('m1', 100, 1000, 900, 500, 5000, 4500, 50.5, 7.5, 1000, 7.6, 5000),
  ('m2', 200, 800, 600, 1000, 4000, 3000, 40.1, 7.0, 900, 6.9, 7000),
  ('m3', 300, NULL, -300, 1500, NULL, -1500, 1.0, 0, 0, NULL, NULL);
INSERT INTO dim_genres VALUES ('g1', 'Horror'), ('g2', 'Comedy');
INSERT INTO bridge_movie_genre VALUES ('m1', 'g1'), ('m2', 'g2'), ('m3', 'g1');
INSERT INTO dim_people VALUES ('p1', 'Ana Atriz', 'Ator'), ('p2', 'Davi Diretor', 'Diretor');
INSERT INTO bridge_movie_person VALUES ('m1', 'p1'), ('m1', 'p2'), ('m2', 'p1');
INSERT INTO dim_companies VALUES ('c1', 'Estúdio X');
INSERT INTO bridge_movie_company VALUES ('m1', 'c1'), ('m2', 'c1');
INSERT INTO movie_reviews (id, sk_movie_review_id, sk_movie_id, name, rating, text, created_at)
  VALUES (1, 'r1', 'm1', 'Rafa', 9.0, 'Ótimo', '2026-01-01');
INSERT INTO dim_reviews VALUES ('m1', 'm1', 1, 9.0);
"""


@pytest.fixture(autouse=True)
def sem_rede(monkeypatch):
    """Qualquer tentativa de requisição HTTP real faz o teste falhar."""

    def bloquear(*args, **kwargs):
        raise AssertionError("Teste tentou acessar a rede (isso gastaria cota do OpenRouter).")

    monkeypatch.setattr(httpx.Client, "send", bloquear)
    monkeypatch.setattr(httpx.AsyncClient, "send", bloquear)


@pytest.fixture
def banco(tmp_path):
    """Um cinerocket.db em miniatura, com o mesmo esquema e poucas linhas."""
    caminho = tmp_path / "mini.db"
    conn = sqlite3.connect(caminho)
    conn.executescript(ESQUEMA)
    conn.commit()
    conn.close()
    return caminho
