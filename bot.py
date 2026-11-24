"""Telegram-бот с агентным RAG: инструменты, цитаты и персистентная история."""

import logging
import re
import threading

import openai
import telebot
from dotenv import load_dotenv

import agent as agent_module
import config
import rag

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

config.require_secrets()

bot = telebot.TeleBot(config.BOT_TOKEN)

HELP_TEXT = (
    "Я RAG-агент: ищу ответы в локальной базе знаний и в интернете.\n\n"
    "Просто задайте вопрос. Если факт есть в базе, я приведу фрагменты как источники; "
    "для свежих данных могу обратиться к веб-поиску.\n\n"
    f"Ограничения: до {config.INPUT_MAX_CHARS} символов в запросе и не больше "
    f"{config.INPUT_RATE_LIMIT} сообщений за {int(config.INPUT_RATE_WINDOW)} с.\n\n"
    "Команды:\n"
    "/reset — очистить историю диалога\n"
    "/status — показать состояние базы знаний\n"
    "/help — эта справка"
)

_chat_locks: dict[int, threading.Lock] = {}
_chat_locks_guard = threading.Lock()


def _chat_lock(chat_id: int) -> threading.Lock:
    with _chat_locks_guard:
        lock = _chat_locks.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            _chat_locks[chat_id] = lock
        return lock


class TypingIndicator:
    """Держит статус «печатает», пока агент думает дольше одного действия."""

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                bot.send_chat_action(self.chat_id, "typing")
            except Exception:
                pass
            self._stop.wait(4.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()


def clean_markdown(text: str) -> str:
    # Сохраняем URL из markdown-ссылок, чтобы источники оставались доступны.
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1: \2", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s", "", text)
    text = re.sub(r"`+", "", text)
    return text.strip()


def split_message(text: str, limit: int = 3900) -> list[str]:
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def send_long_message(message, text: str) -> None:
    for index, part in enumerate(split_message(text)):
        if index == 0:
            bot.reply_to(message, part)
        else:
            bot.send_message(message.chat.id, part)


def format_status(info: dict) -> str:
    return (
        "База знаний:\n"
        f"• файл: {info['file']} ({info['file_size']} байт)\n"
        f"• чанков в индексе: {info['chunks']}\n"
        f"• модель эмбеддингов: {info['embedding_model']}\n"
        f"• sha256: {info['sha256']}\n"
        f"• проиндексировано: {info['indexed_at'] or 'неизвестно'}\n\n"
        f"Модель ответа: {config.MODEL_NAME}\n"
        f"Веб-поиск: {'включён' if config.WEB_SEARCH_ENABLED else 'выключен'}\n"
        f"Чтение страниц: {'включено' if config.WEB_READ_ENABLED else 'выключено'}\n"
        f"Перезапись запроса: {'включена' if config.QUERY_REWRITE_ENABLED else 'выключена'}\n"
        f"Входной слой: до {config.INPUT_MAX_CHARS} символов, "
        f"{config.INPUT_RATE_LIMIT} сообщений / {int(config.INPUT_RATE_WINDOW)} с"
    )


# --- Инициализация RAG ---
print("Запуск инициализации RAG (базы знаний)...")
vectorstore = rag.build_or_load_db()
agent = agent_module.build_agent(vectorstore)


@bot.message_handler(commands=["start", "help"])
def handle_help(message):
    bot.reply_to(message, HELP_TEXT)


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    agent.reset(message.chat.id)
    bot.reply_to(message, "История диалога очищена.")


@bot.message_handler(commands=["status"])
def handle_status(message):
    bot.reply_to(message, format_status(agent.status()))


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    if not message.text:
        bot.reply_to(message, "Пока я умею отвечать только на текстовые сообщения.")
        return

    user_message = message.text.strip()
    if not user_message:
        return
    if user_message.startswith("/"):
        bot.reply_to(message, "Неизвестная команда. Список команд: /help")
        return

    chat_id = message.chat.id
    user_key = getattr(message.from_user, "id", None) or chat_id
    logger.info("User (%s): %s", chat_id, user_message)

    try:
        with _chat_lock(chat_id), TypingIndicator(chat_id):
            reply = agent.reply(chat_id, user_message, user_key=user_key)

        if reply.blocked:
            logger.info(
                "Запрос отклонён входным слоем (chat_id=%s, reason=%s)",
                chat_id,
                reply.block_reason,
            )
            send_long_message(message, reply.text)
            return

        clean_response = clean_markdown(reply.text)
        logger.info(
            "Bot (%s): tools=%s steps=%s elapsed=%.1fs",
            chat_id,
            [event["name"] for event in reply.tool_events],
            reply.steps,
            reply.elapsed,
        )
        send_long_message(message, clean_response)
    except openai.RateLimitError:
        logger.warning("Лимит OpenRouter исчерпан (chat_id=%s)", chat_id)
        bot.reply_to(
            message,
            "Модель сейчас перегружена или исчерпаны лимиты OpenRouter. "
            "Подождите минуту-другую и повторите вопрос.",
        )
    except openai.APIError as exc:
        logger.error("Ошибка API LLM (chat_id=%s): %r", chat_id, exc)
        bot.reply_to(message, "Сервис моделей временно недоступен. Попробуйте позже.")
    except Exception:
        logger.exception("Ошибка обработки сообщения для chat_id=%s", chat_id)
        bot.reply_to(message, "Ошибка при генерации ответа. Попробуйте позже.")


if __name__ == "__main__":
    print("Бот запущен и готов к работе!")
    bot.infinity_polling(timeout=60, long_polling_timeout=30)
