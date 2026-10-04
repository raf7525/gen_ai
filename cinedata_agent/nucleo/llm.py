"""Único ponto do projeto que chama a API do OpenRouter.

Tudo o que protege a cota diária (50 requisições, falhas incluídas) mora aqui:
- cache em disco: a mesma conversa nunca é enviada duas vezes;
- guarda de cota: consulta /api/v1/key (não gasta cota) e um contador local, e recusa
  chamar quando sobra só a reserva;
- fallback: 429 de provedor lotado passa para o próximo modelo, e o que falhou fica alguns
  minutos fora do rodízio (também na execução seguinte); 429 de cota esgotada para tudo;
- sem retentativas automáticas (max_retries=0) e com intervalo mínimo entre chamadas;
- registro de cada requisição real, inclusive as que falham (.cache/requisicoes.jsonl).
"""

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from cinedata_agent.nucleo.config import Settings

Invocador = Callable[[str, list[BaseMessage], Sequence], AIMessage]
ConsultaCota = Callable[[], dict | None]


class ErroLLM(Exception):
    """Base dos erros que impedem obter uma resposta do modelo."""


class CotaEsgotada(ErroLLM):
    """A cota diária de modelos gratuitos acabou. Volta às 21h (horário de Brasília)."""


class CotaNaReserva(ErroLLM):
    """Sobrou só a reserva configurada (RESERVA_COTA); o agente não gasta o resto."""


class ModelosIndisponiveis(ErroLLM):
    """Todos os modelos tentados falharam (lotados, fora do ar ou com erro)."""


class ForaDoCache(ErroLLM):
    """No modo só-cache, a resposta não estava guardada; a API não foi chamada."""


@dataclass
class RespostaLLM:
    mensagem: AIMessage
    modelo: str
    do_cache: bool
    requisicoes: int | None = None  # requisições reais gastas, falhas do fallback incluídas

    def __post_init__(self):
        if self.requisicoes is None:
            self.requisicoes = 0 if self.do_cache else 1


def chave_cache(mensagens: Sequence[BaseMessage], ferramentas: Sequence) -> str:
    """Hash do conteúdo da conversa e das ferramentas, sem ids gerados a cada execução."""
    conversa = []
    for m in mensagens:
        item = {"tipo": m.type, "conteudo": m.content}
        if isinstance(m, AIMessage) and m.tool_calls:
            item["ferramentas"] = [[c["name"], c["args"]] for c in m.tool_calls]
        conversa.append(item)
    schemas = [convert_to_openai_tool(f) for f in ferramentas]
    bruto = json.dumps([conversa, schemas], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(bruto.encode()).hexdigest()


class CacheLLM:
    def __init__(self, caminho: Path):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(caminho, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS respostas "
            "(chave TEXT PRIMARY KEY, modelo TEXT, mensagem TEXT, criado_em TEXT)"
        )

    def obter(self, chave: str) -> RespostaLLM | None:
        linha = self._conn.execute(
            "SELECT modelo, mensagem FROM respostas WHERE chave = ?", (chave,)
        ).fetchone()
        if linha is None:
            return None
        dados = json.loads(linha[1])
        mensagem = AIMessage(
            content=dados["content"],
            tool_calls=[{**c, "type": "tool_call"} for c in dados["tool_calls"]],
        )
        return RespostaLLM(mensagem, linha[0], do_cache=True)

    def guardar(self, chave: str, resposta: RespostaLLM) -> None:
        m = resposta.mensagem
        dados = {
            "content": m.content,
            "tool_calls": [
                {"name": c["name"], "args": c["args"], "id": c["id"]} for c in m.tool_calls
            ],
        }
        self._conn.execute(
            "INSERT OR REPLACE INTO respostas VALUES (?, ?, ?, ?)",
            (chave, resposta.modelo, json.dumps(dados, ensure_ascii=False), _agora_utc()),
        )
        self._conn.commit()


class ContadorUso:
    """Requisições reais feitas hoje (dia UTC: a cota do OpenRouter zera à meia-noite UTC)."""

    def __init__(self, caminho: Path):
        self._caminho = caminho
        caminho.parent.mkdir(parents=True, exist_ok=True)

    def hoje(self) -> int:
        try:
            dados = json.loads(self._caminho.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return 0
        return dados["requisicoes"] if dados.get("dia_utc") == _dia_utc() else 0

    def incrementar(self) -> None:
        dados = {"dia_utc": _dia_utc(), "requisicoes": self.hoje() + 1}
        self._caminho.write_text(json.dumps(dados))


class RegistroRequisicoes:
    """Uma linha JSON por requisição real (sucesso ou falha): modelo, resultado, tempo e tokens.
    Serve para auditar onde a cota foi gasta (`cinedata --cota` mostra as de hoje)."""

    def __init__(self, caminho: Path):
        self._caminho = caminho
        caminho.parent.mkdir(parents=True, exist_ok=True)

    def registrar(self, **dados) -> None:
        linha = {"dia_utc": _dia_utc(), "hora": f"{datetime.now():%H:%M:%S}", **dados}
        with self._caminho.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")

    def hoje(self) -> list[dict]:
        try:
            linhas = self._caminho.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        registros = [json.loads(linha) for linha in linhas if linha.strip()]
        return [r for r in registros if r.get("dia_utc") == _dia_utc()]


class PausasModelos:
    """Modelo que falhou fica alguns minutos fora do rodízio. O arquivo faz a pausa valer também
    na execução seguinte (cada `cinedata "pergunta"` é um processo novo), para não gastar outra
    requisição num provedor que acabou de recusar."""

    def __init__(self, caminho: Path, agora: Callable[[], float] = time.time):
        self._caminho = caminho
        self._agora = agora
        caminho.parent.mkdir(parents=True, exist_ok=True)

    def _ler(self) -> dict[str, float]:
        try:
            return json.loads(self._caminho.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def pausar(self, modelo: str, segundos: float) -> None:
        agora = self._agora()
        ativas = {m: ate for m, ate in self._ler().items() if ate > agora}
        ativas[modelo] = agora + segundos
        self._caminho.write_text(json.dumps(ativas))

    def pausados(self) -> set[str]:
        agora = self._agora()
        return {m for m, ate in self._ler().items() if ate > agora}


def _tipo(mensagem: AIMessage) -> str:
    """'ferramenta' (pediu para executar SQL), 'texto' (respondeu) ou 'vazia' (nenhum dos dois)."""
    if mensagem.tool_calls:
        return "ferramenta"
    conteudo = mensagem.content
    if isinstance(conteudo, list):
        conteudo = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in conteudo)
    return "texto" if (conteudo or "").strip() else "vazia"


def _contexto(mensagens: Sequence[BaseMessage]) -> dict:
    """Pergunta e etapa da requisição, para o registro."""
    pergunta = next((m.content for m in reversed(mensagens) if m.type == "human"), "")
    etapa = "gerar SQL" if mensagens and mensagens[-1].type == "human" else "após a ferramenta"
    return {"pergunta": str(pergunta)[:100], "etapa": etapa}


def _dia_utc() -> str:
    return datetime.now(UTC).date().isoformat()


def _agora_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def consultar_cota_openrouter(settings: Settings) -> dict | None:
    """GET /api/v1/key: devolve {"used", "limit", "remaining"}. Não consome cota."""
    try:
        resp = httpx.get(
            f"{settings.base_url}/key",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["data"].get("free_model_daily_requests")
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def invocador_openrouter(settings: Settings) -> Invocador:
    from langchain_openai import ChatOpenAI

    clientes: dict[str, ChatOpenAI] = {}

    def invocar(modelo: str, mensagens: list[BaseMessage], ferramentas: Sequence) -> AIMessage:
        if modelo not in clientes:
            clientes[modelo] = ChatOpenAI(
                model=modelo,
                base_url=settings.base_url,
                api_key=settings.openrouter_api_key,
                temperature=0,
                max_retries=0,  # o padrão (2) transformaria uma falha em 3 requisições
                timeout=settings.timeout_llm_s,
            )
        cliente = clientes[modelo]
        if ferramentas:
            return cliente.bind_tools(ferramentas).invoke(mensagens)
        return cliente.invoke(mensagens)

    return invocar


class LLM:
    def __init__(
        self,
        settings: Settings,
        invocar: Invocador | None = None,
        consultar_cota: ConsultaCota | None = None,
        relogio: Callable[[], float] = time.monotonic,
        dormir: Callable[[float], None] = time.sleep,
        so_cache: bool = False,
        agora: Callable[[], float] = time.time,
    ):
        self.settings = settings
        self.so_cache = so_cache  # True: nunca chama a API (0 requisições garantidas)
        pasta = settings.caminho(settings.cache_dir)
        self.cache = CacheLLM(pasta / "llm.sqlite")
        self.contador = ContadorUso(pasta / "uso.json")
        self.registro = RegistroRequisicoes(pasta / "requisicoes.jsonl")
        self.pausas = PausasModelos(pasta / "pausas.json", agora)
        self._invocar = invocar or invocador_openrouter(settings)
        self._consultar_cota = consultar_cota or (lambda: consultar_cota_openrouter(settings))
        self._relogio = relogio
        self._dormir = dormir
        self._ultima_chamada: float | None = None

    def chamar(self, mensagens: list[BaseMessage], ferramentas: Sequence = ()) -> RespostaLLM:
        chave = chave_cache(mensagens, ferramentas)
        if (guardada := self.cache.obter(chave)) is not None:
            return guardada
        if self.so_cache:
            raise ForaDoCache("Resposta fora do cache; no modo só-cache a API não é chamada.")

        self._garantir_cota()
        contexto = _contexto(mensagens)
        erros = []
        gastas = 0
        for modelo in self._modelos_disponiveis()[: self.settings.max_tentativas_por_chamada]:
            self._respeitar_intervalo()
            self.contador.incrementar()  # conta antes: requisições que falham também gastam cota
            gastas += 1
            inicio = self._relogio()
            try:
                mensagem = self._invocar(modelo, mensagens, ferramentas)
            except openai.RateLimitError as e:
                acabou = self._cota_acabou(e)
                resultado = "429: cota do dia" if acabou else "429: provedor lotado"
                self._registrar(contexto, modelo, inicio, resultado)
                if acabou:
                    raise CotaEsgotada(
                        "A cota diária de modelos gratuitos acabou; ela volta às 21h (Brasília)."
                    ) from e
                self.pausas.pausar(modelo, self.settings.pausa_modelo_s)
                erros.append(f"{modelo}: provedor lotado (429)")
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                self._registrar(contexto, modelo, inicio, "chave recusada")
                raise ErroLLM(f"Chave do OpenRouter recusada: {e}") from e
            except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
                self._registrar(contexto, modelo, inicio, type(e).__name__)
                self.pausas.pausar(modelo, self.settings.pausa_modelo_s)
                erros.append(f"{modelo}: {type(e).__name__}")
            else:
                vazia = _tipo(mensagem) == "vazia"
                self._registrar(contexto, modelo, inicio, "vazia" if vazia else "ok", mensagem)
                if vazia:
                    # Sem SQL e sem texto: não vai para o cache (a pergunta ficaria presa a uma
                    # resposta inútil) e o próximo modelo tenta.
                    self.pausas.pausar(modelo, self.settings.pausa_modelo_s)
                    erros.append(f"{modelo}: resposta vazia")
                    continue
                resposta = RespostaLLM(mensagem, modelo, do_cache=False, requisicoes=gastas)
                self.cache.guardar(chave, resposta)
                return resposta
        detalhes = "; ".join(erros or ["sem modelos configurados"])
        raise ModelosIndisponiveis(f"Nenhum modelo respondeu: {detalhes}")

    def _registrar(
        self,
        contexto: dict,
        modelo: str,
        inicio: float,
        resultado: str,
        mensagem: AIMessage | None = None,
    ) -> None:
        dados = contexto | {
            "modelo": modelo,
            "resultado": resultado,
            "segundos": round(self._relogio() - inicio, 1),
        }
        if mensagem is not None:
            uso = mensagem.usage_metadata or {}
            dados |= {
                "tipo": _tipo(mensagem),
                "tokens_entrada": uso.get("input_tokens"),
                "tokens_saida": uso.get("output_tokens"),
                "fim": mensagem.response_metadata.get("finish_reason"),
            }
        self.registro.registrar(**dados)

    def cota_restante(self) -> int:
        """Estimativa conservadora: o menor entre o que a API informa e o contador local
        (o contador da API demora a atualizar)."""
        local = self.settings.limite_diario - self.contador.hoje()
        api = self._consultar_cota()
        if api and api.get("remaining") is not None:
            return min(int(api["remaining"]), local)
        return local

    def _garantir_cota(self) -> None:
        restante = self.cota_restante()
        if restante <= 0:
            raise CotaEsgotada("A cota diária acabou; ela volta às 21h (Brasília).")
        if restante <= self.settings.reserva_cota:
            raise CotaNaReserva(
                f"Restam {restante} requisições hoje, que são a reserva (RESERVA_COTA="
                f"{self.settings.reserva_cota}). Diminua a reserva no .env para usá-las."
            )

    def _cota_acabou(self, erro: openai.RateLimitError) -> bool:
        texto = str(erro).lower()
        if "per-day" in texto or "per day" in texto:
            return True
        if "provider" in texto or "upstream" in texto:
            return False
        api = self._consultar_cota()
        return bool(api) and api.get("remaining") == 0

    def _modelos_disponiveis(self) -> list[str]:
        modelos = self.settings.lista_modelos
        pausados = self.pausas.pausados()
        livres = [m for m in modelos if m not in pausados]
        return livres or modelos  # todos pausados: tenta mesmo assim, na ordem configurada

    def _respeitar_intervalo(self) -> None:
        if self._ultima_chamada is not None:
            espera = self.settings.intervalo_min_s - (self._relogio() - self._ultima_chamada)
            if espera > 0:
                self._dormir(espera)
        self._ultima_chamada = self._relogio()
