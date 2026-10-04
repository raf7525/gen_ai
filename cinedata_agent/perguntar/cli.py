"""Linha de comando.

cinedata "Top 10 filmes com maior receita em R$"
cinedata --chat                  # conversa com memória
cinedata --cota                  # requisições restantes hoje (não gasta cota)
cinedata --so-sql "pergunta"     # só gera e executa o SQL (1 chamada)
"""

import argparse
import re
import sys
import uuid
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from cinedata_agent.nucleo.config import get_settings
from cinedata_agent.nucleo.formatacao import formatar
from cinedata_agent.nucleo.llm import ErroLLM
from cinedata_agent.perguntar.charts import desenhar
from cinedata_agent.perguntar.context import sem_acento
from cinedata_agent.perguntar.graph import Agente, Resposta

LINHAS_NA_TABELA = 15


def renderizar(console: Console, r: Resposta, cota: int | None = None) -> None:
    if r.resposta:
        console.print(Panel(Markdown(r.resposta), title=r.pergunta, border_style="blue"))
    if r.df is not None and not r.df.empty:
        tabela = Table(show_lines=False, header_style="bold")
        for coluna in r.df.columns:
            tabela.add_column(coluna)
        for _, linha in r.df.head(LINHAS_NA_TABELA).iterrows():
            tabela.add_row(*(formatar(c, linha[c]) for c in r.df.columns))
        mais = len(r.df) - LINHAS_NA_TABELA
        legenda = f"+{mais} linhas" if mais > 0 else None
        tabela.caption = f"{legenda} (resultado truncado)" if r.truncado else legenda
        console.print(tabela)
    if r.sql:
        console.print(Syntax(r.sql, "sql", word_wrap=True, theme="ansi_dark"))
    if r.premissas:
        console.print("[bold]Premissas:[/bold]")
        for premissa in r.premissas:
            console.print(f"  • {premissa}")
    origem = f" ({r.cache_hits} do cache)" if r.cache_hits else ""
    modelos = ", ".join(dict.fromkeys(r.modelos)) or "—"
    rodape = (
        f"modelo: {modelos} · chamadas ao LLM: {r.chamadas}{origem} · "
        f"requisições gastas: {r.requisicoes}"
    )
    if cota is not None:
        rodape += f" · cota restante hoje: ~{cota}"
    console.print(f"[dim]{rodape}[/dim]")


def tabela_de_requisicoes(registros: list[dict]) -> Table:
    """As requisições reais de hoje (o cache não aparece: ele não gasta cota)."""
    tabela = Table(title="Requisições de hoje", header_style="bold", title_justify="left")
    for coluna in ("hora", "modelo", "etapa", "resultado", "tempo", "tokens", "pergunta"):
        tabela.add_column(coluna)
    for r in registros:
        tokens = "—"
        if r.get("tokens_entrada") is not None:
            tokens = f"{r['tokens_entrada']} → {r.get('tokens_saida')}"
        resultado = r["resultado"] + (f" ({r['tipo']})" if r.get("tipo") else "")
        tabela.add_row(
            r["hora"],
            r["modelo"].split("/")[-1],
            r["etapa"],
            resultado,
            f"{r['segundos']:.0f}s",
            tokens,
            r["pergunta"][:45],
        )
    return tabela


def salvar_grafico(r: Resposta, pasta) -> str | None:
    figura = desenhar(r.df, r.grafico, r.pergunta)
    if figura is None:
        return None
    pasta.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", sem_acento(r.pergunta))[:50].strip("-")
    caminho = pasta / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.png"
    figura.savefig(caminho, dpi=120)
    return str(caminho)


def responder(agente: Agente, console: Console, pergunta: str, conversa: str, args) -> bool:
    modo = "so_sql" if args.so_sql else "completo"
    try:
        with console.status("Consultando..."):
            r = agente.perguntar(pergunta, conversa=conversa, modo=modo)
    except ErroLLM as e:
        console.print(f"[red]{e}[/red]")
        return False
    renderizar(console, r, cota=agente.llm.cota_restante() if r.requisicoes else None)
    if not args.sem_grafico and (
        caminho := salvar_grafico(r, agente.settings.caminho(agente.settings.saidas_dir))
    ):
        console.print(f"[dim]gráfico salvo em {caminho}[/dim]")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cinedata", description=__doc__.split("\n")[0])
    parser.add_argument("pergunta", nargs="*", help="pergunta em linguagem natural")
    parser.add_argument("--chat", action="store_true", help="conversa com memória")
    parser.add_argument("--cota", action="store_true", help="mostra a cota restante")
    parser.add_argument("--so-sql", action="store_true", help="não gera a resposta em texto")
    parser.add_argument("--sem-grafico", action="store_true", help="não salva gráficos")
    args = parser.parse_args(argv)
    console = Console()
    settings = get_settings()

    if not settings.openrouter_api_key:
        console.print("[red]Defina OPENROUTER_API_KEY no arquivo .env (veja .env.example).[/red]")
        return 1
    try:
        agente = Agente(settings)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    if args.cota:
        restante = agente.llm.cota_restante()
        console.print(
            f"Restam ~{restante} requisições hoje (reserva: {settings.reserva_cota}). "
            "A cota zera às 21h (horário de Brasília)."
        )
        if registros := agente.llm.registro.hoje():
            console.print(tabela_de_requisicoes(registros))
        return 0

    if args.chat:
        conversa = uuid.uuid4().hex
        console.print("[dim]Pergunte sobre o catálogo de filmes. 'sair' para encerrar.[/dim]")
        while True:
            try:
                pergunta = console.input("[bold green]você>[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if pergunta.lower() in {"", "sair", "exit", "quit"}:
                break
            responder(agente, console, pergunta, conversa, args)
        return 0

    if not args.pergunta:
        parser.print_help()
        return 1
    ok = responder(agente, console, " ".join(args.pergunta), uuid.uuid4().hex, args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
