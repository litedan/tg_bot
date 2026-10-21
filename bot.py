import os
import re
import logging
import telebot
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Импортируем наш модуль с RAG
import rag

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- 1. Настройки LLM ---
OPENROUTER_BASE = "http://127.0.0.1:11434/v1"
MODEL_NAME = "gemma2:2b"

llm = ChatOpenAI(
    openai_api_key="fake_key",
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
    request_timeout=120, # Увеличенный таймаут для RAG
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Не задан BOT_TOKEN. Добавьте токен Telegram-бота в .env.")
bot = telebot.TeleBot(BOT_TOKEN)

# История диалогов
chat_history = {}

# Функция очистки Markdown
def clean_markdown(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'_+', '', text)
    return text

# --- 2. Инициализация RAG ---
print("Запуск инициализации RAG (базы знаний)...")
vectorstore = rag.build_or_load_db()

# --- 3. Обработчик сообщений ---
@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    if not message.text:
        bot.reply_to(message, "Пока я умею отвечать только на текстовые сообщения.")
        return

    chat_id = message.chat.id
    user_message = message.text.strip()
    if not user_message:
        return

    try:
        # Инициализация истории для нового пользователя
        if chat_id not in chat_history:
            chat_history[chat_id] = []

        print(f"User ({chat_id}): {user_message}")
        print(f"History length: {len(chat_history[chat_id])}")

        # --- 4. Поиск контекста в RAG ---
        # Функция из rag.py ищет нужные куски текста и вставляет их в системный промпт
        system_prompt_text = rag.get_context_and_prompt(user_message, vectorstore)

        # Формируем финальный список: Системный промпт + История диалога
        messages_to_send = (
            [SystemMessage(content=system_prompt_text)]
            + chat_history[chat_id]
            + [HumanMessage(content=user_message)]
        )

        # --- 5. Отправка в LLM ---
        response = llm.invoke(messages_to_send).content
        if not isinstance(response, str) or not response.strip():
            raise ValueError("LLM вернула пустой ответ")
        
        # Очистка от случайного Markdown
        clean_response = clean_markdown(response)

        # Добавляем в историю только завершившийся обмен, чтобы ошибка LLM
        # не оставляла там вопрос без ответа.
        chat_history[chat_id].append(HumanMessage(content=user_message))
        chat_history[chat_id].append(AIMessage(content=clean_response))
        
        # Снова проверяем лимит истории
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        print(f"Bot: {clean_response}")

        # Отправляем ответ в Telegram
        bot.reply_to(message, clean_response)

    except Exception as e:
        logger.exception("Ошибка обработки сообщения для chat_id=%s", chat_id)
        bot.reply_to(message, "Ошибка при генерации ответа. Попробуйте позже.")

# --- Запуск ---
if __name__ == "__main__":
    print("Бот запущен и готов к работе!")
    bot.infinity_polling(timeout=60, long_polling_timeout=30)
