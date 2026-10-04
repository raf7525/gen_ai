import pytest

from cinedata_agent.nucleo.formatacao import compacto, formatar


@pytest.mark.parametrize(
    ("coluna", "valor", "esperado"),
    [
        ("receita_brl", 12390136500.54, "R$ 12.390.136.500,54"),
        ("lucro_medio", -1500.5, "R$ -1.500,50"),
        ("orcamento_usd", 888000000, "US$ 888.000.000,00"),
        ("margem_pct", 74.67, "74,67%"),
        ("margem_lucro_pct", 99.79, "99,79%"),
        ("ano_lancamento", 2023, "2023"),
        ("filmes", 28086, "28.086"),
        ("nota_media_imdb", 6.3456, "6,35"),
        ("titulo", "Barbie", "Barbie"),
        ("receita_brl", None, "—"),
    ],
)
def test_formata_no_padrao_brasileiro(coluna, valor, esperado):
    assert formatar(coluna, valor) == esperado


def test_formato_compacto_para_eixos():
    assert compacto(12.39e9, "R$ ") == "R$ 12,39 bi"
    assert compacto(3_400_000) == "3,40 mi"
    assert compacto(950) == "950,00"
