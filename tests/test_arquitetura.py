"""Regras das fatias verticais: o núcleo não conhece as fatias, e uma fatia só usa a API
pública da outra (o que o __init__.py dela exporta)."""

import ast
from pathlib import Path

import pytest

PACOTE = Path(__file__).resolve().parent.parent / "cinedata_agent"


def importacoes(pasta: str) -> list[tuple[str, str]]:
    """(arquivo, módulo importado) de todos os imports internos da pasta."""
    achadas = []
    for arquivo in (PACOTE / pasta).glob("*.py"):
        for no in ast.walk(ast.parse(arquivo.read_text(encoding="utf-8"))):
            if isinstance(no, ast.ImportFrom) and (no.module or "").startswith("cinedata_agent"):
                achadas.append((arquivo.name, no.module))
            elif isinstance(no, ast.Import):
                achadas += [
                    (arquivo.name, a.name) for a in no.names if a.name.startswith("cinedata_agent")
                ]
    return achadas


def test_nucleo_nao_depende_das_fatias():
    proibidas = [
        (a, m) for a, m in importacoes("nucleo") if not m.startswith("cinedata_agent.nucleo")
    ]
    assert proibidas == []


@pytest.mark.parametrize(("fatia", "outra"), [("avaliar", "perguntar"), ("perguntar", "avaliar")])
def test_fatia_usa_so_a_api_publica_da_outra(fatia, outra):
    internas = [(a, m) for a, m in importacoes(fatia) if m.startswith(f"cinedata_agent.{outra}.")]
    assert internas == []
