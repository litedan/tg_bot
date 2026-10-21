import telebot
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv() 


# Замени на свой токен от @BotFather
TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(
        message.chat.id,
        f"Привет, {message.from_user.first_name}! 👋\n"
        f"Я простейший бот. Напиши что-нибудь, и я повторю."
    )


@bot.message_handler(commands=['help'])
def help_message(message):
    bot.send_message(
        message.chat.id,
        "Доступные команды:\n"
        "/start — приветствие\n"
        "/help — помощь\n"
        "Или просто отправь любой текст — я повторю его."
    )


@bot.message_handler(content_types=['text'])
def echo_message(message):
    bot.reply_to(message, f"Ты написал: {message.text}")


if __name__ == "__main__":
    print("Бот запущен...")
    bot.infinity_polling()