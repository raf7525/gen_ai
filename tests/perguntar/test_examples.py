"""Os exemplos do prompt precisam continuar válidos no banco real (pulado sem o banco)."""

import pytest

from cinedata_agent.nucleo.config import get_settings
from cinedata_agent.nucleo.db import executar
from cinedata_agent.nucleo.guardrails import validar_sql
from cinedata_agent.perguntar.context import exemplos

BANCO = get_settings().caminho(get_settings().db_path)


@pytest.mark.skipif(not BANCO.exists(), reason="cinerocket.db não está em data/")
@pytest.mark.parametrize("exemplo", exemplos(), ids=lambda e: e["pergunta"][:40])
def test_exemplo_passa_nos_guardrails_e_retorna_linhas(exemplo):
    resultado = executar(validar_sql(exemplo["sql"]), BANCO)
    assert len(resultado.df) > 0
    assert not any(c.startswith("sk_") for c in resultado.df.columns)
