"""Единая конфигурация проекта (переменные окружения и константы)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- OpenRouter / LLM ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
# Фиксированная tool-capable модель вместо openrouter/free:
# роутер free выбирает модель случайно, и агентный цикл становится нестабильным.
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
# Резервная модель: если основная отдаёт 429/5xx после всех ретраев,
# ответ достраивается на ней. Пусто = отключено.
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "").strip()
REQUEST_TIMEOUT = _env_int("LLM_REQUEST_TIMEOUT", 120)
MAX_AGENT_STEPS = _env_int("AGENT_MAX_STEPS", 4)
HISTORY_LIMIT = _env_int("HISTORY_LIMIT", 8)
APP_NAME = os.getenv("APP_NAME", "tg-rag-agent")

# --- Ретраи запросов к LLM (429/перегрузка провайдера/сеть) ---
LLM_RETRIES = _env_int("LLM_RETRIES", 3)
LLM_RETRY_BASE_DELAY = _env_float("LLM_RETRY_BASE_DELAY", 2.0)
LLM_RETRY_MAX_DELAY = _env_float("LLM_RETRY_MAX_DELAY", 20.0)

# --- Telegram ---
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# --- RAG / индекс ---
CHROMA_DIR = str(BASE_DIR / "chroma_db")
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
# v2 — коллекция с E5 passage-префиксами; при смене модели эмбеддингов меняйте имя.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ru_knowledge_base_v2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
RETRIEVAL_CANDIDATES = _env_int("RETRIEVAL_CANDIDATES", 12)
RERANKED_DOCUMENTS = _env_int("RERANKED_DOCUMENTS", 4)
# Минимальная релевантность после cross-encoder; ниже — считаем мусором.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))

# --- Веб-поиск ---
WEB_SEARCH_ENABLED = _env_bool("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_MAX_RESULTS = _env_int("WEB_SEARCH_MAX_RESULTS", 5)
WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "ru-ru")

# --- Чтение веб-страниц (инструмент read_page) ---
WEB_READ_ENABLED = _env_bool("WEB_READ_ENABLED", True)
WEB_READ_TIMEOUT = _env_int("WEB_READ_TIMEOUT", 15)
WEB_READ_MAX_CHARS = _env_int("WEB_READ_MAX_CHARS", 6000)
WEB_READ_MAX_BYTES = _env_int("WEB_READ_MAX_BYTES", 2_000_000)

# --- Входной слой (input_layer.py) ---
# Максимальная длина запроса после нормализации.
INPUT_MAX_CHARS = max(1, _env_int("INPUT_MAX_CHARS", 4000))
# Скользящее окно лимита частоты: не больше N сообщений за W секунд на пользователя.
INPUT_RATE_LIMIT = max(1, _env_int("INPUT_RATE_LIMIT", 10))
INPUT_RATE_WINDOW = max(1.0, _env_float("INPUT_RATE_WINDOW", 60.0))
# Если невидимых символов больше доли и сообщение длиннее порога — блокируем.
INPUT_INVISIBLE_MAX_RATIO = min(
    max(_env_float("INPUT_INVISIBLE_MAX_RATIO", 0.05), 0.0), 1.0
)
INPUT_INVISIBLE_MIN_CHARS = max(1, _env_int("INPUT_INVISIBLE_MIN_CHARS", 100))

# --- Перезапись поискового запроса с учётом диалога ---
QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", True)
QUERY_REWRITE_MESSAGES = _env_int("QUERY_REWRITE_MESSAGES", 6)
QUERY_REWRITE_TIMEOUT = _env_int("QUERY_REWRITE_TIMEOUT", 30)

# --- Состояние бота ---
STATE_DIR = BASE_DIR / "state"
HISTORY_FILE = STATE_DIR / "history.json"


def require_secrets() -> None:
    missing = []
    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if missing:
        raise RuntimeError(
            "Не заданы переменные окружения: " + ", ".join(missing) + ". Добавьте их в .env."
        )
