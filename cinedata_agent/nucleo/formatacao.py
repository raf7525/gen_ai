"""Formatação de números no padrão brasileiro, a partir do nome da coluna."""

import math
import numbers

MONETARIAS = ("receita", "lucro", "orcamento", "faturamento", "bilheteria", "prejuizo", "_brl")


def tipo_da_coluna(coluna: str) -> str:
    nome = coluna.lower()
    # Porcentagem primeiro: "margem_lucro_pct" cita lucro, mas não é valor em R$.
    if "margem" in nome or "pct" in nome or "percent" in nome:
        return "pct"
    if nome.endswith("_usd") or "dolar" in nome:
        return "usd"
    if any(chave in nome for chave in MONETARIAS):
        return "brl"
    return "numero"


def _br(valor: float, casas: int) -> str:
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def compacto(valor: float, prefixo: str = "") -> str:
    """1.239.014.000 -> 'R$ 1,24 bi' (para eixos de gráfico e resumos)."""
    for limite, sufixo in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
        if abs(valor) >= limite:
            return f"{prefixo}{_br(valor / limite, 2)}{sufixo}"
    return f"{prefixo}{_br(valor, 2)}"


def formatar(coluna: str, valor) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    if isinstance(valor, bool) or not isinstance(valor, numbers.Number):
        return str(valor)
    tipo = tipo_da_coluna(coluna)
    if tipo == "brl":
        return f"R$ {_br(valor, 2)}"
    if tipo == "usd":
        return f"US$ {_br(valor, 2)}"
    if tipo == "pct":
        return f"{_br(valor, 2)}%"
    if "ano" in coluna.lower() and float(valor).is_integer():
        return str(int(valor))  # anos sem separador de milhar
    if float(valor).is_integer():
        return _br(valor, 0)
    return _br(valor, 2)
