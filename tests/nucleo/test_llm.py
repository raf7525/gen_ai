import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cinedata_agent.nucleo.config import Settings
from cinedata_agent.nucleo.llm import (
    LLM,
    CotaEsgotada,
    CotaNaReserva,
    ErroLLM,
    ForaDoCache,
    ModelosIndisponiveis,
    PausasModelos,
    chave_cache,
)

MENSAGENS = [SystemMessage("sistema"), HumanMessage("Top 10 filmes por receita")]


def erro_http(classe, status, mensagem):
    resposta = httpx.Response(status, request=httpx.Request("POST", "https://openrouter.ai"))
    return classe(mensagem, response=resposta, body=None)


class InvocadorFalso:
    """Devolve (ou levanta) respostas roteirizadas e registra cada chamada."""

    def __init__(self, *roteiro):
        self.roteiro = list(roteiro)
        self.chamadas = []

    def __call__(self, modelo, mensagens, ferramentas):
        self.chamadas.append(modelo)
        item = self.roteiro.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def criar_llm(tmp_path, invocador, restante=50, **ajustes):
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        modelos="modelo-a,modelo-b,modelo-c",
        intervalo_min_s=0,
        reserva_cota=5,
        **ajustes,
    )
    return LLM(settings, invocar=invocador, consultar_cota=lambda: {"remaining": restante})


def test_resposta_vai_para_o_cache_e_segunda_chamada_nao_gasta_cota(tmp_path):
    invocador = InvocadorFalso(AIMessage("oi"))
    llm = criar_llm(tmp_path, invocador)

    primeira = llm.chamar(MENSAGENS)
    segunda = llm.chamar(MENSAGENS)

    assert (primeira.do_cache, segunda.do_cache) == (False, True)
    assert segunda.mensagem.content == "oi"
    assert invocador.chamadas == ["modelo-a"]
    assert llm.contador.hoje() == 1


def test_cache_preserva_chamadas_de_ferramenta(tmp_path):
    chamada = {"name": "executar_sql", "args": {"sql": "SELECT 1"}, "id": "c1"}
    llm = criar_llm(tmp_path, InvocadorFalso(AIMessage("", tool_calls=[chamada])))
    llm.chamar(MENSAGENS)

    guardada = llm.chamar(MENSAGENS)

    assert guardada.mensagem.tool_calls[0]["args"] == {"sql": "SELECT 1"}


def test_chave_do_cache_ignora_ids_gerados_a_cada_execucao():
    a = [HumanMessage("p", id="uuid-1"), AIMessage("r", id="uuid-2")]
    b = [HumanMessage("p", id="uuid-3"), AIMessage("r", id="uuid-4")]
    assert chave_cache(a, []) == chave_cache(b, [])
    assert chave_cache(a, []) != chave_cache([HumanMessage("outra")], [])
    assert chave_cache([ToolMessage("x", tool_call_id="1")], []) == chave_cache(
        [ToolMessage("x", tool_call_id="2")], []
    )


def test_provedor_lotado_passa_para_o_proximo_modelo(tmp_path):
    lotado = erro_http(openai.RateLimitError, 429, "Provider returned error: upstream busy")
    invocador = InvocadorFalso(lotado, AIMessage("resposta do b"))
    llm = criar_llm(tmp_path, invocador)

    resposta = llm.chamar(MENSAGENS)

    assert resposta.modelo == "modelo-b"
    assert invocador.chamadas == ["modelo-a", "modelo-b"]
    assert llm.contador.hoje() == 2  # a falha também contou


def test_modelo_lotado_fica_fora_do_rodizio_nas_proximas_chamadas(tmp_path):
    lotado = erro_http(openai.RateLimitError, 429, "upstream provider busy")
    invocador = InvocadorFalso(lotado, AIMessage("1"), AIMessage("2"))
    llm = criar_llm(tmp_path, invocador)

    llm.chamar(MENSAGENS)
    llm.chamar([HumanMessage("outra pergunta")])

    assert invocador.chamadas == ["modelo-a", "modelo-b", "modelo-b"]


def test_cota_diaria_esgotada_para_tudo_sem_tentar_outros_modelos(tmp_path):
    esgotada = erro_http(openai.RateLimitError, 429, "Rate limit exceeded: free-models-per-day")
    invocador = InvocadorFalso(esgotada, AIMessage("nunca chega aqui"))
    llm = criar_llm(tmp_path, invocador)

    with pytest.raises(CotaEsgotada):
        llm.chamar(MENSAGENS)
    assert invocador.chamadas == ["modelo-a"]


def test_429_ambiguo_consulta_a_api_para_decidir(tmp_path):
    ambiguo = erro_http(openai.RateLimitError, 429, "Rate limit exceeded")
    llm = criar_llm(tmp_path, InvocadorFalso(ambiguo), restante=50)
    llm._consultar_cota = lambda: {"remaining": 0}  # a API confirma: acabou

    with pytest.raises(CotaEsgotada):
        llm.chamar(MENSAGENS)


def test_reserva_bloqueia_antes_de_gastar(tmp_path):
    invocador = InvocadorFalso(AIMessage("x"))
    llm = criar_llm(tmp_path, invocador, restante=5)

    with pytest.raises(CotaNaReserva, match="RESERVA_COTA"):
        llm.chamar(MENSAGENS)
    assert invocador.chamadas == []


def test_cota_restante_usa_o_contador_local_quando_a_api_atrasa(tmp_path):
    llm = criar_llm(tmp_path, InvocadorFalso(), restante=50)
    for _ in range(10):
        llm.contador.incrementar()
    assert llm.cota_restante() == 40


def test_cota_restante_usa_contador_local_sem_api(tmp_path):
    llm = criar_llm(tmp_path, InvocadorFalso())
    llm._consultar_cota = lambda: None
    assert llm.cota_restante() == 50


def test_limita_tentativas_por_chamada(tmp_path):
    fora = erro_http(openai.InternalServerError, 500, "erro interno")
    invocador = InvocadorFalso(fora, fora, AIMessage("não deveria tentar"))
    llm = criar_llm(tmp_path, invocador, max_tentativas_por_chamada=2)

    with pytest.raises(ModelosIndisponiveis):
        llm.chamar(MENSAGENS)
    assert invocador.chamadas == ["modelo-a", "modelo-b"]


def test_chave_invalida_nao_tenta_outros_modelos(tmp_path):
    recusada = erro_http(openai.AuthenticationError, 401, "No auth credentials found")
    invocador = InvocadorFalso(recusada)
    llm = criar_llm(tmp_path, invocador)

    with pytest.raises(ErroLLM, match="Chave do OpenRouter"):
        llm.chamar(MENSAGENS)
    assert invocador.chamadas == ["modelo-a"]


def test_respeita_intervalo_minimo_entre_chamadas(tmp_path):
    esperas = []
    settings = Settings(_env_file=None, cache_dir=tmp_path, modelos="m", intervalo_min_s=3.5)
    llm = LLM(
        settings,
        invocar=InvocadorFalso(AIMessage("1"), AIMessage("2")),
        consultar_cota=lambda: {"remaining": 50},
        relogio=lambda: 100.0,  # o relógio não anda: a segunda chamada precisa esperar tudo
        dormir=esperas.append,
    )
    llm.chamar([HumanMessage("a")])
    llm.chamar([HumanMessage("b")])
    assert esperas == [3.5]


def test_resposta_vazia_nao_vai_para_o_cache_e_o_proximo_modelo_tenta(tmp_path):
    invocador = InvocadorFalso(AIMessage(""), AIMessage("resposta do b"))
    llm = criar_llm(tmp_path, invocador)

    resposta = llm.chamar(MENSAGENS)
    de_novo = llm.chamar(MENSAGENS)

    assert (resposta.modelo, resposta.requisicoes) == ("modelo-b", 2)
    assert (de_novo.do_cache, de_novo.mensagem.content) == (True, "resposta do b")
    assert invocador.chamadas == ["modelo-a", "modelo-b"]


def test_registra_cada_requisicao_inclusive_as_que_falham(tmp_path):
    lotado = erro_http(openai.RateLimitError, 429, "Provider returned error: upstream busy")
    resposta = AIMessage(
        "ok", usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940}
    )
    llm = criar_llm(tmp_path, InvocadorFalso(lotado, resposta))

    llm.chamar(MENSAGENS)
    llm.chamar(MENSAGENS)  # do cache: não é requisição, não entra no registro

    registros = llm.registro.hoje()
    assert [(r["modelo"], r["resultado"]) for r in registros] == [
        ("modelo-a", "429: provedor lotado"),
        ("modelo-b", "ok"),
    ]
    assert registros[1]["etapa"] == "gerar SQL"
    assert registros[1]["pergunta"] == "Top 10 filmes por receita"
    assert (registros[1]["tokens_entrada"], registros[1]["tokens_saida"]) == (900, 40)


def test_modo_so_cache_nunca_chama_a_api(tmp_path):
    invocador = InvocadorFalso(AIMessage("não deveria chegar aqui"))
    llm = criar_llm(tmp_path, invocador)
    llm.so_cache = True

    with pytest.raises(ForaDoCache):
        llm.chamar(MENSAGENS)
    assert invocador.chamadas == []
    assert llm.contador.hoje() == 0


def test_pausa_vale_tambem_na_proxima_execucao(tmp_path):
    lotado = erro_http(openai.RateLimitError, 429, "Provider returned error: upstream busy")
    criar_llm(tmp_path, InvocadorFalso(lotado, AIMessage("b"))).chamar(MENSAGENS)

    # Um processo novo (outro LLM, mesma pasta de cache) já começa pelo modelo-b.
    invocador = InvocadorFalso(AIMessage("direto no b"))
    criar_llm(tmp_path, invocador).chamar([HumanMessage("outra pergunta")])

    assert invocador.chamadas == ["modelo-b"]


def test_erro_de_servidor_tambem_pausa_o_modelo(tmp_path):
    fora = erro_http(openai.InternalServerError, 500, "erro interno")
    invocador = InvocadorFalso(fora, AIMessage("1"), AIMessage("2"))
    llm = criar_llm(tmp_path, invocador)

    llm.chamar(MENSAGENS)
    llm.chamar([HumanMessage("outra pergunta")])

    assert invocador.chamadas == ["modelo-a", "modelo-b", "modelo-b"]


def test_pausa_expira_depois_do_prazo(tmp_path):
    relogio = [1000.0]
    pausas = PausasModelos(tmp_path / "pausas.json", agora=lambda: relogio[0])
    pausas.pausar("modelo-a", 300)
    assert pausas.pausados() == {"modelo-a"}

    relogio[0] += 301
    assert pausas.pausados() == set()
