"""Configuração lida do .env (veja .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parents[2]  # cinedata_agent/nucleo/config.py -> raiz do projeto


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=RAIZ / ".env", extra="ignore")

    openrouter_api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    modelos: str = "qwen/qwen3.8-27b:free,nvidia/nemotron-3.5-lightning:free,openrouter/free"

    # Cota diária dos modelos gratuitos: as falhas também contam.
    limite_diario: int = 50
    reserva_cota: int = 5
    intervalo_min_s: float = 3.5
    pausa_modelo_s: float = 300  # modelo que falhou fica fora do rodízio por 5 min
    max_tentativas_por_chamada: int = 2
    timeout_llm_s: float = 90

    # Agente
    max_chamadas_por_pergunta: int = 3
    turnos_memoria: int = 3

    # Banco
    db_path: Path = Path("data/cinerocket.db")
    timeout_sql_s: float = 60
    limite_padrao: int = 100
    limite_max: int = 1000

    cache_dir: Path = Path(".cache")
    saidas_dir: Path = Path("saidas")

    @property
    def lista_modelos(self) -> list[str]:
        return [m.strip() for m in self.modelos.split(",") if m.strip()]

    def caminho(self, p: Path) -> Path:
        """Caminhos relativos valem a partir da raiz do projeto, não da pasta atual."""
        return p if p.is_absolute() else RAIZ / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
