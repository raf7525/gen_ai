"""Acesso somente leitura à camada Gold (SQLite)."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TABELAS = frozenset(
    {
        "dim_movies",
        "fact_movies_performance",
        "dim_genres",
        "dim_people",
        "dim_companies",
        "dim_reviews",
        "movie_reviews",
        "bridge_movie_genre",
        "bridge_movie_person",
        "bridge_movie_company",
    }
)


class ErroConsulta(Exception):
    """A consulta falhou no SQLite (sintaxe, coluna inexistente, timeout...)."""


@dataclass
class Resultado:
    df: pd.DataFrame
    truncado: bool  # havia mais linhas do que o limite lido


def conectar(caminho: Path) -> sqlite3.Connection:
    """Abre o banco em modo somente leitura: nenhuma escrita é possível, nem por engano."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Banco não encontrado em {caminho}. Baixe o cinerocket.db para a pasta data/."
        )
    conn = sqlite3.connect(f"file:{caminho}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn


def executar(sql: str, caminho: Path, timeout_s: float = 60, max_linhas: int = 1000) -> Resultado:
    """Executa um SELECT já validado e devolve no máximo `max_linhas` linhas."""
    conn = conectar(caminho)
    prazo = time.monotonic() + timeout_s
    # Chamado a cada N instruções da VM do SQLite; retornar 1 interrompe a consulta.
    conn.set_progress_handler(lambda: int(time.monotonic() > prazo), 10_000)
    try:
        cur = conn.execute(sql)
        colunas = [c[0] for c in cur.description or []]
        linhas = cur.fetchmany(max_linhas + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise ErroConsulta(
                f"A consulta passou de {timeout_s:.0f} s e foi interrompida. "
                "Simplifique: filtre antes de juntar tabelas grandes."
            ) from e
        raise ErroConsulta(str(e)) from e
    except sqlite3.Error as e:
        raise ErroConsulta(str(e)) from e
    finally:
        conn.close()
    truncado = len(linhas) > max_linhas
    return Resultado(pd.DataFrame(linhas[:max_linhas], columns=colunas), truncado)
