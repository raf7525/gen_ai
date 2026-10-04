"""Grafo do agente (LangGraph): uma versão enxuta do CHESS.

    START → recuperar_informacao → selecionar_esquema → agente ⇄ testar_sql → finalizar → END

Só o nó `agente` chama o LLM: 1 vez para gerar o SQL (via tool calling) e 1 para responder.
As demais etapas do CHESS (Information Retriever, Schema Selector e Unit Tester) são Python.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Annotated, Literal, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from cinedata_agent.nucleo.config import Settings, get_settings
from cinedata_agent.nucleo.db import ErroConsulta, executar
from cinedata_agent.nucleo.guardrails import SQLBloqueado, validar_sql
from cinedata_agent.nucleo.llm import LLM
from cinedata_agent.perguntar.context import Informacao, normalizar, recuperar
from cinedata_agent.perguntar.prompts import montar_sistema

LINHAS_PARA_O_MODELO = 20


class ArgsExecutarSQL(BaseModel):
    sql: str = Field(description="Uma única consulta SELECT no dialeto SQLite.")
    premissas: list[str] = Field(
        default_factory=list,
        description="Decisões tomadas, em frases curtas (ex.: 'só filmes com receita informada').",
    )
    grafico: Literal["barra", "linha", "nenhum"] = Field(
        default="nenhum",
        description="barra = ranking/comparação; linha = evolução no tempo; nenhum = valor único.",
    )
    eixo_x: str | None = Field(default=None, description="Coluna do resultado para o eixo x.")
    eixo_y: str | None = Field(default=None, description="Coluna numérica para o eixo y.")


@tool("executar_sql", args_schema=ArgsExecutarSQL)
def executar_sql(**_: object) -> str:
    """Executa uma consulta SELECT na camada Gold (SQLite, somente leitura) e devolve as
    primeiras linhas do resultado, ou o erro para você corrigir a consulta."""
    raise NotImplementedError("Executada pelo nó testar_sql do grafo.")


class Estado(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    pergunta: str
    modo: Literal["completo", "so_sql"]
    informacao: dict
    sistema: str
    chamadas: int
    requisicoes: int
    cache_hits: int
    modelos: list[str]
    tentativas_sql: int
    sql: str | None
    premissas: list[str]
    grafico: dict | None
    resultado: dict | None
    erro: str | None
    resposta: str


@dataclass
class Resposta:
    pergunta: str
    resposta: str
    sql: str | None = None
    premissas: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None
    truncado: bool = False
    grafico: dict | None = None
    erro: str | None = None
    chamadas: int = 0
    requisicoes: int = 0
    cache_hits: int = 0
    modelos: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- utilidades


def texto_da_mensagem(mensagem: AnyMessage) -> str:
    """Conteúdo em texto puro, sem o raciocínio que alguns modelos devolvem em <think>."""
    conteudo = mensagem.content
    if isinstance(conteudo, list):
        conteudo = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in conteudo)
    return re.sub(r"<think>.*?</think>", "", conteudo or "", flags=re.S).strip()


def _sql_em_texto(texto: str) -> dict | None:
    """Alguns modelos escrevem a chamada em texto em vez de usar tool calling. Aproveitar
    evita gastar outra requisição só para pedir o formato certo."""
    bloco = re.search(r"```sql\s*(.+?)```", texto, flags=re.S | re.I)
    if bloco and re.match(r"\s*(SELECT|WITH)\b", bloco.group(1), flags=re.I):
        return {"sql": bloco.group(1).strip()}
    candidato = re.search(r"\{.*\}", texto, flags=re.S)
    if candidato:
        try:
            dados = json.loads(candidato.group(0))
        except json.JSONDecodeError:
            return None
        if isinstance(dados, dict):
            args = dados.get("arguments") or dados.get("parameters") or dados
            if isinstance(args, dict) and isinstance(args.get("sql"), str):
                return args
    return None


def _registros(df: pd.DataFrame) -> dict:
    limpo = df.astype(object).where(df.notna(), None)
    return {"colunas": list(df.columns), "linhas": limpo.values.tolist()}


def _df(resultado: dict | None) -> pd.DataFrame | None:
    if resultado is None:
        return None
    return pd.DataFrame(resultado["linhas"], columns=resultado["colunas"])


def _tabela_para_o_modelo(df: pd.DataFrame, truncado: bool) -> str:
    amostra = df.head(LINHAS_PARA_O_MODELO).map(
        lambda v: v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v
    )
    total = f"{len(df)}{'+' if truncado else ''}"
    rodape = f"\n\n({total} linhas no total; mostrando {len(amostra)})"
    return amostra.to_markdown(index=False) + rodape


def _checar_regras(sql_original: str) -> str | None:
    """Unit Tester: regras de negócio que uma consulta precisa respeitar."""
    usa_lucro = re.search(r"lucro_(brl|usd)", sql_original, flags=re.I)
    filtra_receita = re.search(
        r"receita_(brl|usd)\s*(IS\s+NOT\s+NULL|>|>=|<>|!=)", sql_original, flags=re.I
    )
    if usa_lucro and not filtra_receita:
        return (
            "A consulta usa lucro sem filtrar filmes com receita informada. Sem receita, o lucro "
            "vira −orçamento e distorce o resultado: adicione WHERE receita_brl IS NOT NULL."
        )
    return None


def janela(mensagens: list[AnyMessage], turnos: int) -> list[AnyMessage]:
    """Memória enxuta: o turno atual inteiro e, dos `turnos` anteriores, só a pergunta,
    a resposta e o SQL usado (os resultados antigos saem do prompt)."""
    blocos: list[list[AnyMessage]] = []
    for m in mensagens:
        if isinstance(m, HumanMessage) or not blocos:
            blocos.append([])
        blocos[-1].append(m)
    resumidas: list[AnyMessage] = []
    for bloco in blocos[:-1][-turnos:] if turnos else []:
        pergunta = next((m for m in bloco if isinstance(m, HumanMessage)), None)
        if pergunta is None:
            continue
        sqls = [
            c["args"].get("sql", "")
            for m in bloco
            if isinstance(m, AIMessage)
            for c in m.tool_calls
        ]
        respostas = [
            texto_da_mensagem(m) for m in bloco if isinstance(m, AIMessage) and not m.tool_calls
        ]
        partes = [r for r in respostas[-1:] if r]
        if sqls:
            partes.append(f"SQL usado: {sqls[-1]}")
        resumidas += [
            HumanMessage(pergunta.content),
            AIMessage("\n\n".join(partes) or "(sem resposta)"),
        ]
    return resumidas + blocos[-1] if blocos else []


# ---------------------------------------------------------------- grafo


def construir_grafo(llm: LLM, settings: Settings) -> StateGraph:
    banco = settings.caminho(settings.db_path)

    def recuperar_informacao(estado: Estado) -> dict:
        anteriores = [m.content for m in estado["messages"][:-1] if isinstance(m, HumanMessage)]
        info = recuperar(estado["pergunta"], historico=anteriores[-settings.turnos_memoria :])
        return {"informacao": asdict(info)}

    def selecionar_esquema(estado: Estado) -> dict:
        return {"sistema": montar_sistema(Informacao(**estado["informacao"]))}

    def agente(estado: Estado) -> dict:
        mensagens = [SystemMessage(estado["sistema"])]
        mensagens += janela(estado["messages"], settings.turnos_memoria)
        resposta = llm.chamar(mensagens, [executar_sql])
        mensagem = resposta.mensagem
        if not mensagem.tool_calls and estado.get("tentativas_sql", 0) == 0:
            args = _sql_em_texto(texto_da_mensagem(mensagem))
            if args:
                chamada = {"name": "executar_sql", "args": args, "id": "sql-em-texto"}
                mensagem = AIMessage("", tool_calls=[chamada])
        return {
            "messages": [mensagem],
            "chamadas": estado.get("chamadas", 0) + 1,
            "requisicoes": estado.get("requisicoes", 0) + resposta.requisicoes,
            "cache_hits": estado.get("cache_hits", 0) + (1 if resposta.do_cache else 0),
            "modelos": [*estado.get("modelos", []), resposta.modelo],
        }

    def testar_sql(estado: Estado) -> dict:
        chamada, *extras = estado["messages"][-1].tool_calls
        tentativas = estado.get("tentativas_sql", 0) + 1
        atualizacao: dict = {"tentativas_sql": tentativas}
        args = chamada["args"] if chamada["name"] == "executar_sql" else {}
        try:
            sql = validar_sql(args.get("sql", ""), settings.limite_padrao, settings.limite_max)
            if aviso := _checar_regras(args["sql"]):
                raise ErroConsulta(aviso)
            resultado = executar(sql, banco, settings.timeout_sql_s, settings.limite_max)
            df = resultado.df.drop(columns=[c for c in resultado.df.columns if c.startswith("sk_")])
        except (SQLBloqueado, ErroConsulta) as e:
            prefixo = "bloqueado" if isinstance(e, SQLBloqueado) else "erro"
            atualizacao["erro"] = f"{prefixo}: {e}"
            conteudo = f"ERRO ({prefixo}): {e}\nCorrija a consulta e chame executar_sql de novo."
        else:
            atualizacao |= {
                "sql": sql,
                "premissas": list(args.get("premissas") or []),
                "grafico": {
                    "tipo": args.get("grafico", "nenhum"),
                    "x": args.get("eixo_x"),
                    "y": args.get("eixo_y"),
                },
                "resultado": _registros(df) | {"truncado": resultado.truncado},
                "erro": None if len(df) else "vazio",
            }
            if df.empty:
                conteudo = (
                    "A consulta rodou, mas não retornou linhas. Se isso não for esperado, revise "
                    "os filtros (gêneros em inglês, nomes exatos, anos). Se for, diga ao usuário."
                )
            else:
                conteudo = _tabela_para_o_modelo(df, resultado.truncado)
        mensagens = [ToolMessage(conteudo, tool_call_id=chamada["id"])]
        mensagens += [
            ToolMessage("Ignorada: envie uma consulta por vez.", tool_call_id=c["id"])
            for c in extras
        ]
        return atualizacao | {"messages": mensagens}

    def finalizar(estado: Estado) -> dict:
        ultima = estado["messages"][-1]
        texto = ""
        if isinstance(ultima, AIMessage) and not ultima.tool_calls:
            texto = texto_da_mensagem(ultima)  # pode sobrar vazio se o modelo só "pensou"
        if texto:
            resposta = texto
        elif estado.get("modo") == "so_sql" and estado.get("sql"):
            resposta = ""
        elif estado.get("erro") and estado["erro"] != "vazio":
            resposta = (
                "Não consegui montar uma consulta válida para essa pergunta. "
                f"Último erro: {estado['erro']}. Tente reformular."
            )
        elif estado.get("resultado"):
            resposta = "Consulta executada; veja a tabela abaixo."
        else:
            resposta = "Não consegui responder a essa pergunta. Tente reformular."
        return {"resposta": resposta}

    def depois_do_agente(estado: Estado) -> str:
        ultima = estado["messages"][-1]
        return "testar_sql" if isinstance(ultima, AIMessage) and ultima.tool_calls else "finalizar"

    def depois_do_teste(estado: Estado) -> str:
        if estado.get("chamadas", 0) >= settings.max_chamadas_por_pergunta:
            return "finalizar"
        if estado.get("erro"):
            # No máximo uma revisão: na segunda falha, desiste sem gastar mais requisições.
            if estado["tentativas_sql"] >= 2 and estado["erro"] != "vazio":
                return "finalizar"
            if estado.get("modo") == "so_sql" and estado["tentativas_sql"] >= 2:
                return "finalizar"
            return "agente"
        return "finalizar" if estado.get("modo") == "so_sql" else "agente"

    grafo = StateGraph(Estado)
    grafo.add_node("recuperar_informacao", recuperar_informacao)
    grafo.add_node("selecionar_esquema", selecionar_esquema)
    grafo.add_node("agente", agente)
    grafo.add_node("testar_sql", testar_sql)
    grafo.add_node("finalizar", finalizar)
    grafo.add_edge(START, "recuperar_informacao")
    grafo.add_edge("recuperar_informacao", "selecionar_esquema")
    grafo.add_edge("selecionar_esquema", "agente")
    grafo.add_conditional_edges("agente", depois_do_agente, ["testar_sql", "finalizar"])
    grafo.add_conditional_edges("testar_sql", depois_do_teste, ["agente", "finalizar"])
    grafo.add_edge("finalizar", END)
    return grafo


class Agente:
    """Interface simples para a CLI, o notebook e a avaliação."""

    def __init__(self, settings: Settings | None = None, llm: LLM | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or LLM(self.settings)
        self.grafo = construir_grafo(self.llm, self.settings).compile(checkpointer=InMemorySaver())

    def perguntar(
        self,
        pergunta: str,
        conversa: str = "padrao",
        modo: Literal["completo", "so_sql"] = "completo",
    ) -> Resposta:
        pergunta = normalizar(pergunta)
        entrada: Estado = {
            "messages": [HumanMessage(pergunta)],
            "pergunta": pergunta,
            "modo": modo,
            "chamadas": 0,
            "requisicoes": 0,
            "cache_hits": 0,
            "modelos": [],
            "tentativas_sql": 0,
            "sql": None,
            "premissas": [],
            "grafico": None,
            "resultado": None,
            "erro": None,
            "resposta": "",
        }
        final = self.grafo.invoke(entrada, {"configurable": {"thread_id": conversa}})
        resultado = final.get("resultado")
        return Resposta(
            pergunta=pergunta,
            resposta=final.get("resposta", ""),
            sql=final.get("sql"),
            premissas=final.get("premissas", []),
            df=_df(resultado),
            truncado=bool(resultado and resultado.get("truncado")),
            grafico=final.get("grafico"),
            erro=final.get("erro"),
            chamadas=final.get("chamadas", 0),
            requisicoes=final.get("requisicoes", 0),
            cache_hits=final.get("cache_hits", 0),
            modelos=final.get("modelos", []),
        )
