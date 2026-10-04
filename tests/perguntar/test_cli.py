import pandas as pd
from rich.console import Console

from cinedata_agent.perguntar import cli
from cinedata_agent.perguntar.charts import desenhar, escolher
from cinedata_agent.perguntar.graph import Resposta

RANKING = pd.DataFrame(
    {
        "titulo": ["Avatar", "Avatar", "Barbie"],
        "ano_lancamento": [2022, 2009, 2023],
        "receita_brl": [12.39e9, 5.1e9, 6.85e9],
    }
)
POR_ANO = pd.DataFrame(
    {"ano_lancamento": [2016, 2017, 2018], "nota_media_imdb": [6.34, 6.34, 6.27]}
)


def test_usa_a_sugestao_do_modelo_quando_valida():
    sugestao = {"tipo": "barra", "x": "titulo", "y": "receita_brl"}
    assert escolher(RANKING, sugestao) == ("barra", "titulo", "receita_brl")


def test_ignora_sugestao_com_coluna_inexistente_e_usa_heuristica():
    sugestao = {"tipo": "barra", "x": "filme", "y": "receita"}
    assert escolher(RANKING, sugestao) == ("barra", "titulo", "receita_brl")
    assert escolher(POR_ANO, None) == ("linha", "ano_lancamento", "nota_media_imdb")


def test_sem_grafico_para_valor_unico_ou_quando_o_modelo_dispensa():
    assert escolher(pd.DataFrame({"filmes": [843]}), None) is None
    assert escolher(RANKING, {"tipo": "nenhum"}) is None
    assert escolher(None, None) is None


def test_desenha_barras_sem_fundir_titulos_repetidos():
    figura = desenhar(RANKING, None, "Maiores receitas")
    rotulos = [t.get_text() for t in figura.axes[0].get_yticklabels()]
    assert rotulos == ["Avatar (2022)", "Avatar (2009)", "Barbie (2023)"]
    assert len(figura.axes[0].patches) == 3


def test_desenha_linha_por_ano():
    figura = desenhar(POR_ANO, {"tipo": "linha", "x": "ano_lancamento", "y": "nota_media_imdb"})
    assert len(figura.axes[0].lines) == 1


def test_linha_nao_liga_anos_que_faltam():
    com_lacuna = pd.DataFrame(
        {"ano_lancamento": [2026, 2027, 2029], "nota_media_imdb": [7.5, 6.4, 3.8]}
    )
    figura = desenhar(com_lacuna, {"tipo": "linha", "x": "ano_lancamento", "y": "nota_media_imdb"})
    anos, notas = figura.axes[0].lines[0].get_data()
    assert list(anos) == [2026, 2027, 2028, 2029]
    assert pd.isna(notas[2])


def test_barras_mostram_generos_em_portugues():
    df = pd.DataFrame({"nome_genero": ["Horror", "Science Fiction"], "margem_pct": [74.67, 68.18]})
    figura = desenhar(df, {"tipo": "barra", "x": "nome_genero", "y": "margem_pct"})
    rotulos = [t.get_text() for t in figura.axes[0].get_yticklabels()]
    assert rotulos == ["Terror", "Ficção científica"]


def test_renderiza_resposta_completa():
    console = Console(record=True, width=120)
    resposta = Resposta(
        pergunta="Maiores receitas",
        resposta="Avatar lidera.",
        sql="SELECT titulo FROM dim_movies LIMIT 10",
        premissas=["só filmes com receita informada"],
        df=RANKING,
        chamadas=2,
        requisicoes=1,
        cache_hits=1,
        modelos=["nemotron"],
    )
    cli.renderizar(console, resposta, cota=37)
    saida = console.export_text()
    for trecho in [
        "Avatar lidera.",
        "R$ 12.390.000.000,00",
        "SELECT titulo",
        "só filmes com",
        "chamadas ao LLM: 2 (1 do cache)",
        "cota restante hoje: ~37",
    ]:
        assert trecho in saida


def test_salva_grafico_em_png(tmp_path):
    resposta = Resposta(pergunta="Maiores receitas?", resposta="", df=RANKING)
    caminho = cli.salvar_grafico(resposta, tmp_path)
    assert caminho.endswith("-maiores-receitas.png")
    assert (tmp_path / caminho.split("/")[-1]).stat().st_size > 0


def test_nome_do_grafico_tira_acentos_sem_cortar_palavras(tmp_path):
    resposta = Resposta(pergunta="Gênero com maior margem", resposta="", df=RANKING)
    assert cli.salvar_grafico(resposta, tmp_path).endswith("-genero-com-maior-margem.png")


def test_sem_chave_explica_como_configurar(monkeypatch, capsys):
    from cinedata_agent.nucleo.config import Settings

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    assert cli.main(["qualquer pergunta"]) == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().out
