"""Fatia "perguntar": recebe uma pergunta em português e devolve resposta, tabela, SQL,
premissas e gráfico. Contém o fluxo CHESS inteiro (contexto, prompt, grafo) e o comando
`cinedata`. Outras fatias usam só o que é exportado aqui."""

from cinedata_agent.perguntar.graph import Agente, Resposta

__all__ = ["Agente", "Resposta"]
