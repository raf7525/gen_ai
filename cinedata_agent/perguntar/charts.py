"""Gráficos automáticos: usa a sugestão do modelo (feita na mesma chamada que gerou o SQL,
sem custo extra de cota) e, se ela não servir, uma heurística pelo formato do resultado."""

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from cinedata_agent.nucleo.formatacao import compacto, tipo_da_coluna
from cinedata_agent.perguntar.context import semantica

MAX_BARRAS = 15


def escolher(df: pd.DataFrame | None, sugestao: dict | None) -> tuple[str, str, str] | None:
    """Devolve (tipo, coluna_x, coluna_y) ou None quando um gráfico não ajuda."""
    if df is None or len(df) < 2:
        return None
    numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if sugestao and sugestao.get("tipo") in ("barra", "linha"):
        x, y = sugestao.get("x"), sugestao.get("y")
        if x in df.columns and y in numericas and x != y:
            return sugestao["tipo"], x, y
    if sugestao and sugestao.get("tipo") == "nenhum":
        return None
    anos = [c for c in numericas if "ano" in c.lower()]
    metricas = [c for c in numericas if c not in anos]
    textos = [c for c in df.columns if c not in numericas]
    if anos and metricas and not textos:
        return "linha", anos[0], metricas[-1]
    if textos and metricas:
        return "barra", textos[0], metricas[-1]
    return None


def desenhar(df: pd.DataFrame | None, sugestao: dict | None, titulo: str = "") -> Figure | None:
    escolha = escolher(df, sugestao)
    if escolha is None:
        return None
    tipo, x, y = escolha
    fig = Figure(figsize=(9, 5), layout="constrained")
    ax = fig.add_subplot()
    if tipo == "linha":
        dados = _com_lacunas(df.dropna(subset=[x]).sort_values(x), x)
        ax.plot(dados[x], dados[y], marker="o", color="#2563eb")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # anos sem casas decimais
        ax.yaxis.set_major_formatter(_formatador(y))
    else:
        dados = df.head(MAX_BARRAS)
        posicoes = range(len(dados))  # posições numéricas: títulos repetidos não se fundem
        ax.barh(posicoes, dados[y], color="#2563eb")
        ax.set_yticks(posicoes, _rotulos(dados, x))
        ax.invert_yaxis()  # o primeiro do ranking fica em cima
        ax.set_xlabel(y)
        ax.xaxis.set_major_formatter(_formatador(y))
    ax.grid(axis="x" if tipo == "barra" else "y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(titulo[:90], fontsize=11)
    return fig


def _com_lacunas(df: pd.DataFrame, x: str) -> pd.DataFrame:
    """Insere os anos (ou outros inteiros) que faltam com valor vazio, para a linha não ligar
    2027 direto a 2029 como se 2028 existisse."""
    valores = df[x]
    if not pd.api.types.is_numeric_dtype(valores) or not (valores % 1 == 0).all():
        return df
    completo = range(int(valores.min()), int(valores.max()) + 1)
    return df.set_index(x).reindex(completo).rename_axis(x).reset_index()


def _rotulos(df: pd.DataFrame, x: str) -> list[str]:
    """Título + ano quando há as duas colunas (há títulos repetidos), cortado em 40 caracteres.
    Gêneros saem em português, como na resposta em texto."""
    rotulos = df[x].astype(str)
    if "genero" in x:
        nomes = {nome: info["pt"] for nome, info in semantica()["generos"].items()}
        rotulos = rotulos.map(lambda v: nomes.get(v, v))
    if x != "ano_lancamento" and "ano_lancamento" in df.columns:
        rotulos = rotulos + " (" + df["ano_lancamento"].astype("Int64").astype(str) + ")"
    return [r if len(r) <= 40 else r[:39] + "…" for r in rotulos]


def _formatador(coluna: str) -> FuncFormatter:
    prefixo = {"brl": "R$ ", "usd": "US$ "}.get(tipo_da_coluna(coluna), "")
    if tipo_da_coluna(coluna) == "pct":
        return FuncFormatter(lambda v, _: f"{v:.0f}%")
    return FuncFormatter(lambda v, _: compacto(v, prefixo))
