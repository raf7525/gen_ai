"""Etapas Information Retriever e Schema Selector do CHESS, sem chamar o LLM.

- recuperar(): normaliza a pergunta, acha gêneros citados, regras de negócio e exemplos parecidos.
- O prompt do sistema é montado em prompts.montar_sistema() a partir do que foi recuperado.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PASTA = Path(__file__).resolve().parent

STOPWORDS = frozenset(
    {
        "a",
        "as",
        "cada",
        "com",
        "como",
        "da",
        "das",
        "de",
        "diga",
        "do",
        "dos",
        "e",
        "em",
        "entre",
        "foi",
        "foram",
        "liste",
        "mais",
        "me",
        "menos",
        "mostre",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "quais",
        "qual",
        "quantas",
        "quanto",
        "quantos",
        "que",
        "sao",
        "se",
        "sem",
        "ser",
        "seu",
        "seus",
        "sobre",
        "sua",
        "suas",
        "tem",
        "ter",
        "teve",
        "um",
        "uma",
        "umas",
        "uns",
    }
)


@dataclass
class Informacao:
    pergunta: str
    regras: list[dict] = field(default_factory=list)
    exemplos: list[dict] = field(default_factory=list)
    generos: list[tuple[str, str]] = field(default_factory=list)  # (nome no banco, nome em PT)


@lru_cache
def semantica() -> dict:
    return yaml.safe_load((PASTA / "semantic.yaml").read_text(encoding="utf-8"))


@lru_cache
def exemplos() -> list[dict]:
    return yaml.safe_load((PASTA / "examples.yaml").read_text(encoding="utf-8"))


def normalizar(pergunta: str) -> str:
    """Remove espaços extras e pontuação final, para variações triviais caírem no mesmo cache."""
    texto = re.sub(r"\s+", " ", pergunta).strip()
    return re.sub(r"[\s?!.;:]+$", "", texto)


def sem_acento(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def palavras(texto: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", sem_acento(texto))


def _radicais(texto: str) -> set[str]:
    """Radical grosseiro (5 letras): 'lançados' e 'lançamento' viram 'lanca'."""
    return {p[:5] for p in palavras(texto) if p not in STOPWORDS and len(p) > 1}


def recuperar(pergunta: str, historico: list[str] | None = None, k: int = 3) -> Informacao:
    """Seleciona o contexto da pergunta. O histórico (perguntas anteriores da conversa)
    entra na seleção de regras, para continuações como "e só os de terror?"."""
    pergunta = normalizar(pergunta)
    textos = [*(historico or []), pergunta]
    tokens = {p for t in textos for p in palavras(t)}
    generos = _generos(tokens)
    return Informacao(
        pergunta=pergunta,
        regras=_regras(tokens, cita_genero=bool(generos)),
        exemplos=_exemplos(pergunta, k),
        generos=generos,
    )


def _regras(tokens: set[str], cita_genero: bool) -> list[dict]:
    selecionadas = []
    for regra in semantica()["regras"]:
        gatilhos = regra.get("gatilhos", [])
        if (
            regra.get("sempre")
            or (cita_genero and regra["id"] == "generos")
            or any(t.startswith(g) for t in tokens for g in gatilhos)
        ):
            selecionadas.append(regra)
    return selecionadas


def _generos(tokens: set[str]) -> list[tuple[str, str]]:
    achados = []
    for nome, info in semantica()["generos"].items():
        if any(termo in tokens for termo in info["termos"]):
            achados.append((nome, info["pt"]))
    return achados


def _exemplos(pergunta: str, k: int) -> list[dict]:
    alvo = _radicais(pergunta)

    def similaridade(exemplo: dict) -> float:
        outro = _radicais(exemplo["pergunta"])
        return len(alvo & outro) / len(alvo | outro) if alvo | outro else 0.0

    ranqueados = sorted(exemplos(), key=similaridade, reverse=True)
    return [e for e in ranqueados[:k] if similaridade(e) > 0]
