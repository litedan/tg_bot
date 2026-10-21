import os
import re
import telebot
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


# ИСПРАВЛЕНО: Добавлен /v1 в конце и заменен localhost на 127.0.0.1
OPENROUTER_BASE = "http://127.0.0.1:11434/v1"
MODEL_NAME = "gemma2:2b"

llm = ChatOpenAI(
    openai_api_key="fake_key",
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
    request_timeout=60, # Если ответа нет 60 секунд, бот выдаст ошибку
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

# Словарь для хранения истории сообщений для каждого пользователя
chat_history = {}

@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        chat_id = message.chat.id
        user_message = message.text

        # Инициализируем историю для нового пользователя
        if chat_id not in chat_history:
            chat_history[chat_id] = []

        # Добавляем вопрос пользователя в историю
        chat_history[chat_id].append(HumanMessage(content=user_message))

        # Оставляем только последние 10 сообщений (5 пар вопрос-ответ)
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        # Отправляем сообщение в LLM с историей
        print(f"User ({chat_id}): {user_message}")
        print(f"History length: {len(chat_history[chat_id])}")

        messages_to_send = chat_history[chat_id]

        # Отправляем запрос
        response = llm.invoke(messages_to_send).content

        # Добавляем очищенный ответ в историю
        chat_history[chat_id].append(AIMessage(content=response))

        # Снова ограничиваем историю последними 10 сообщениями
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        print(f"Bot: {response}")

        # Отвечаем пользователю
        bot.reply_to(message, response)

    except Exception as e:
        # Выводим ошибку в консоль, чтобы было видно, что не так
        print(f"ОШИБКА: {e}")
        error_msg = f"Ошибка при генерации ответа. Попробуйте позже."
        bot.reply_to(message, error_msg)

# Запуск бота
if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling()