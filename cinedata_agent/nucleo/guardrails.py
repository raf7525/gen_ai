"""Validação do SQL gerado pelo modelo antes de executar.

Primeira linha de defesa (a segunda é a conexão somente leitura do db.py):
só passa um único comando de leitura, sobre as tabelas da camada Gold, com LIMIT.
"""

import sqlglot
from sqlglot import exp

from cinedata_agent.nucleo.db import TABELAS

FUNCOES_PROIBIDAS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"})


class SQLBloqueado(Exception):
    """O SQL não passou nos guardrails; a mensagem explica o motivo ao modelo."""


def validar_sql(sql: str, limite_padrao: int = 100, limite_max: int = 1000) -> str:
    """Devolve o SQL pronto para executar, ou levanta SQLBloqueado.

    Se o LIMIT já está adequado, devolve o texto original do modelo (mais legível); senão,
    a versão reescrita pelo sqlglot com o LIMIT corrigido. O texto executado é sempre o mesmo
    que foi analisado aqui."""
    texto = sql.strip().rstrip(";").strip()
    if not texto:
        raise SQLBloqueado("SQL vazio.")
    try:
        comandos = [c for c in sqlglot.parse(texto, read="sqlite") if c is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLBloqueado(f"SQL inválido: {e}") from e

    if len(comandos) != 1:
        raise SQLBloqueado("Envie um único comando SELECT, sem ';' separando comandos.")
    consulta = comandos[0]
    if not isinstance(consulta, exp.Query):
        raise SQLBloqueado(
            "Apenas consultas de leitura (SELECT/WITH) são permitidas; "
            f"recebi um comando do tipo {type(consulta).__name__.upper()}."
        )

    _checar_tabelas(consulta)
    _checar_funcoes(consulta)
    limitada = _aplicar_limite(consulta, limite_padrao, limite_max)
    if limitada is consulta:
        return texto
    if consulta.args.get("limit") is None:
        # Acrescenta o LIMIT ao texto original e confere que a análise continua a mesma.
        candidato = f"{texto}\nLIMIT {limite_padrao}"
        if sqlglot.parse_one(candidato, read="sqlite") == limitada:
            return candidato
    return limitada.sql(dialect="sqlite", pretty=True)


def _checar_tabelas(consulta: exp.Query) -> None:
    ctes = {cte.alias_or_name.lower() for cte in consulta.find_all(exp.CTE)}
    for tabela in consulta.find_all(exp.Table):
        nome = tabela.name.lower()
        if nome in ctes:
            continue
        if tabela.args.get("db"):
            raise SQLBloqueado(f"Não use prefixo de banco ({tabela.sql('sqlite')}).")
        if nome not in TABELAS:
            raise SQLBloqueado(
                f"Tabela desconhecida: {tabela.name}. Use apenas: {', '.join(sorted(TABELAS))}."
            )


def _checar_funcoes(consulta: exp.Query) -> None:
    for funcao in consulta.find_all(exp.Func):
        nome = (funcao.sql_name() if not isinstance(funcao, exp.Anonymous) else funcao.name).lower()
        if nome in FUNCOES_PROIBIDAS:
            raise SQLBloqueado(f"Função não permitida: {nome}.")


def _aplicar_limite(consulta: exp.Query, limite_padrao: int, limite_max: int) -> exp.Query:
    limite = consulta.args.get("limit")
    if limite is None:
        return consulta.copy().limit(limite_padrao)
    valor = limite.expression
    if not (isinstance(valor, exp.Literal) and valor.is_int):
        raise SQLBloqueado("Use um número inteiro no LIMIT.")
    if int(valor.this) > limite_max:
        return consulta.copy().limit(limite_max)
    return consulta
