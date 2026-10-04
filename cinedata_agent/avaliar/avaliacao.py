"""Avaliação do agente por acurácia de execução (a métrica do CHESS e do benchmark BIRD):
o resultado do SQL gerado é comparado com o resultado de um SQL ouro escrito à mão.

    python -m cinedata_agent.avaliar --offline      # valida o SQL ouro (0 requisições)
    python -m cinedata_agent.avaliar                # todas as perguntas, modo só-SQL
    python -m cinedata_agent.avaliar --ids F1,E3    # só algumas
    python -m cinedata_agent.avaliar --completo     # também gera a resposta em texto
    python -m cinedata_agent.avaliar --so-cache     # só o que está no cache (0 requisições)

O modo só-SQL gasta 1 requisição por pergunta (2 se precisar revisar). Repetir uma rodada
com o mesmo prompt sai do cache e não gasta nada.
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from cinedata_agent.nucleo.config import RAIZ, Settings, get_settings
from cinedata_agent.nucleo.db import executar
from cinedata_agent.nucleo.formatacao import formatar
from cinedata_agent.nucleo.guardrails import validar_sql
from cinedata_agent.nucleo.llm import LLM, ErroLLM, ForaDoCache
from cinedata_agent.perguntar import Agente, Resposta

PASTA = RAIZ / "avaliacao"


def carregar(ids: list[str] | None = None, arquivo: Path = PASTA / "perguntas.yaml") -> list[dict]:
    perguntas = yaml.safe_load(arquivo.read_text(encoding="utf-8"))
    if ids:
        perguntas = [p for p in perguntas if p["id"] in ids]
    return perguntas


# ---------------------------------------------------------------- comparação


def valores_iguais(esperado, obtido) -> bool:
    if esperado is None or (isinstance(esperado, float) and math.isnan(esperado)):
        return obtido is None or (isinstance(obtido, float) and math.isnan(obtido))
    if isinstance(esperado, (int, float)) and not isinstance(esperado, bool):
        try:
            valor = float(obtido)
        except (TypeError, ValueError):
            return False
        tolerancia = max(0.01, abs(esperado) * 0.01)
        # Aceita a mesma métrica como fração ou porcentagem (0,95 ou 95%).
        return any(abs(esperado * f - valor) <= tolerancia * f for f in (1, 100, 0.01))
    return str(esperado).strip().lower() == str(obtido).strip().lower()


def _contem(linha_esperada: list, linha_obtida: list) -> bool:
    return all(any(valores_iguais(e, o) for o in linha_obtida) for e in linha_esperada)


def comparar(esperado: pd.DataFrame, obtido: pd.DataFrame | None, cfg: dict) -> float:
    """Fração (0 a 1) das linhas esperadas que aparecem no resultado do agente."""
    if obtido is None or obtido.empty:
        return 0.0
    k = cfg.get("k", 5) or len(esperado)
    alvo = esperado[cfg.get("colunas") or list(esperado.columns)].head(k)
    janela = len(obtido) if not cfg.get("k") else max(2 * k, k + 3)
    linhas = obtido.head(janela).astype(object).values.tolist()

    if cfg.get("modo") == "metrica":
        esperados = alvo.iloc[:, 0].tolist()
        melhor = 0.0
        for coluna in range(obtido.shape[1]):
            valores = [linha[coluna] for linha in linhas[: len(esperados)]]
            iguais = sum(valores_iguais(e, o) for e, o in zip(esperados, valores, strict=False))
            melhor = max(melhor, iguais / len(esperados))
        return melhor

    livres = list(linhas)
    encontradas = 0
    for linha in alvo.astype(object).values.tolist():
        par = next((o for o in livres if _contem(linha, o)), None)
        if par is not None:
            livres.remove(par)  # cada linha do agente casa com uma esperada só
            encontradas += 1
    return encontradas / len(alvo)


def classificar(nota: float) -> str:
    return "acerto" if nota >= 0.8 else "parcial" if nota >= 0.4 else "erro"


def avaliar(pergunta: dict, resposta: Resposta, esperado: pd.DataFrame | None) -> dict:
    tipo = pergunta.get("tipo", "dados")
    if tipo == "fora_do_escopo":
        nota = 1.0 if resposta.sql is None else 0.0
    elif tipo == "injecao":
        bloqueou = (resposta.erro or "").startswith("bloqueado")
        nota = 1.0 if resposta.sql is None or bloqueou else 0.5
    else:
        nota = comparar(esperado, resposta.df, pergunta.get("comparar", {}))
    return {"nota": nota, "classe": classificar(nota)}


# ---------------------------------------------------------------- execução


def resultado_ouro(pergunta: dict, settings: Settings) -> pd.DataFrame | None:
    if "sql_ouro" not in pergunta:
        return None
    sql = validar_sql(pergunta["sql_ouro"], settings.limite_padrao, settings.limite_max)
    return executar(sql, settings.caminho(settings.db_path), settings.timeout_sql_s).df


def validar_offline(perguntas: list[dict], settings: Settings) -> bool:
    ok = True
    for p in perguntas:
        if "sql_ouro" not in p:
            continue
        inicio = time.time()
        try:
            df = resultado_ouro(p, settings)
            colunas = p.get("comparar", {}).get("colunas") or []
            faltando = [c for c in colunas if c not in df.columns]
            situacao = f"{len(df)} linhas" + (f", colunas ausentes: {faltando}" if faltando else "")
            ok &= not faltando and len(df) > 0
        except Exception as e:  # noqa: BLE001 - relatório de validação
            situacao, ok = f"FALHOU: {e}", False
        print(f"{p['id']:4} {time.time() - inicio:5.1f}s  {situacao}")
    return ok


def rodar(
    agente: Agente, perguntas: list[dict], modo: str
) -> tuple[list[dict], str | None, list[str]]:
    resultados, parada, puladas = [], None, []
    for p in perguntas:
        inicio = time.time()
        try:
            esperado = resultado_ouro(p, agente.settings)
            resposta = agente.perguntar(p["pergunta"], conversa=f"avaliacao-{p['id']}", modo=modo)
        except ForaDoCache:
            puladas.append(p["id"])
            print(f"{p['id']:4} fora do cache: pulada (0 requisições)")
            continue
        except ErroLLM as e:
            parada = f"Parou antes de {p['id']}: {e}"
            break
        item = {"pergunta": p, "resposta": resposta, "segundos": time.time() - inicio}
        item |= avaliar(p, resposta, esperado)
        resultados.append(item)
        print(
            f"{p['id']:4} {item['classe']:8} nota={item['nota']:.2f} "
            f"chamadas={resposta.chamadas} requisições={resposta.requisicoes} "
            f"({item['segundos']:.0f}s)"
        )
    return resultados, parada, puladas


def relatorio(
    resultados: list[dict], modo: str, parada: str | None, total: int, puladas: list[str] = ()
) -> str:
    acertos = sum(r["classe"] == "acerto" for r in resultados)
    parciais = sum(r["classe"] == "parcial" for r in resultados)
    requisicoes = sum(r["resposta"].requisicoes for r in resultados)
    chamadas = sum(r["resposta"].chamadas for r in resultados)
    modelos = sorted({m for r in resultados for m in r["resposta"].modelos})
    linhas = [
        f"# Avaliação do agente — {datetime.now():%d/%m/%Y %H:%M}",
        "",
        f"- **Modo:** {modo} · **modelos:** {', '.join(modelos) or '—'}",
        f"- **Acertos:** {acertos}/{len(resultados)} · **parciais:** {parciais}",
        f"- **Chamadas ao LLM:** {chamadas} "
        f"(média {chamadas / max(len(resultados), 1):.2f} por pergunta) · "
        f"**requisições gastas:** {requisicoes} (o resto veio do cache)",
    ]
    if parada:
        linhas.append(f"- **Interrompida:** {parada} ({len(resultados)}/{total} executadas)")
    if puladas:
        linhas.append(f"- **Fora do cache (não avaliadas, modo só-cache):** {', '.join(puladas)}")
    linhas += [
        "",
        "| Id | Categoria | Pergunta | Resultado | Nota | Chamadas | Requisições | Modelo "
        "| Tempo |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in resultados:
        p, resp = r["pergunta"], r["resposta"]
        linhas.append(
            f"| {p['id']} | {p['categoria']} | {p['pergunta']} | {r['classe']} | "
            f"{r['nota']:.2f} | {resp.chamadas} | {resp.requisicoes} | "
            f"{', '.join(dict.fromkeys(resp.modelos)) or '—'} | {r['segundos']:.0f}s |"
        )
    linhas.append("\n## Detalhes\n")
    for r in resultados:
        p, resp = r["pergunta"], r["resposta"]
        linhas += [f"### {p['id']} — {p['pergunta']} ({r['classe']})", ""]
        if resp.resposta:
            linhas += [f"> {resp.resposta.replace(chr(10), ' ')}", ""]
        if resp.premissas:
            linhas += [f"Premissas: {'; '.join(resp.premissas)}", ""]
        linhas += ["SQL gerado:", "```sql", resp.sql or "(nenhum)", "```"]
        if "sql_ouro" in p:
            linhas += [
                "<details><summary>SQL ouro</summary>",
                "",
                "```sql",
                p["sql_ouro"],
                "```",
                "</details>",
            ]
        if resp.df is not None and not resp.df.empty:
            linhas += ["", _tabela(resp.df.head(5))]
        linhas.append("")
    return "\n".join(linhas)


def _tabela(df: pd.DataFrame) -> str:
    """Tabela em Markdown com números no padrão brasileiro (sem notação científica)."""
    return df.apply(lambda col: col.map(lambda v: formatar(col.name, v))).to_markdown(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avalia o agente com o conjunto de perguntas.")
    parser.add_argument("--ids", help="ids separados por vírgula (ex.: F1,E3)")
    parser.add_argument("--offline", action="store_true", help="só valida o SQL ouro")
    parser.add_argument("--completo", action="store_true", help="gera também a resposta")
    parser.add_argument(
        "--so-cache", action="store_true", help="usa só o cache (0 requisições); pula o resto"
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    perguntas = carregar(args.ids.split(",") if args.ids else None)

    if args.offline:
        return 0 if validar_offline(perguntas, settings) else 1

    modo = "completo" if args.completo else "so_sql"
    if args.so_cache:
        agente = Agente(settings, LLM(settings, so_cache=True))
        print("Modo só-cache: nenhuma requisição será feita.")
    else:
        agente = Agente(settings)
        print(f"Cota restante estimada: {agente.llm.cota_restante()}")
    resultados, parada, puladas = rodar(agente, perguntas, modo)
    if parada:
        print(parada)
    texto = relatorio(resultados, modo, parada, len(perguntas), puladas)
    destino = PASTA / "resultados" / f"{datetime.now():%Y-%m-%d-%H%M%S}-{modo}.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    acertos = sum(r["classe"] == "acerto" for r in resultados)
    print(f"\n{acertos}/{len(resultados)} acertos. Relatório: {destino.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
