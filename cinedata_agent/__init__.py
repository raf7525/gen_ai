"""Agente Text-to-SQL da CineData, organizado em fatias verticais.

- perguntar/: pergunta em português -> SQL -> resposta (o fluxo CHESS, do prompt ao terminal)
- avaliar/:   avaliação do agente por acurácia de execução
- nucleo/:    o que as fatias compartilham (configuração, banco, guardrails, LLM, formatação)
"""
