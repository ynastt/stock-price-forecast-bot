import asyncio
import logging
from datetime import datetime
import os
import re
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.filters.command import Command
import yfinance as yf
from forecast_models import StockForecaster
from utils import save_to_log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем токен из переменных окружения
TOKEN = os.getenv('TG_BOT_TOKEN')

if not TOKEN:
    raise ValueError("TG_BOT_TOKEN не найден в переменных окружения. Создайте файл .env")

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: types.Message):
    welcome_msg ="""
    Привет!👋\nЯ твой бот-помощник для анализа и прогнозирования акций на основе временных рядов!\n\nПришли мне тикер компании (например, AAPL, MSFT) и сумму денег для условной инвестиции, и я дам прогноз на 30 дней вперед.
Для начала вам надо выбрать тикер, компании которая вас интересует, например Apple это AAPL, Google это GOOGL.
Полный список тикеров вы можете найти на сайте https://finance.yahoo.com/.
Далее надо ввести команду /analyze \[ТИКЕР] \[СУММА]

*Пример:* 
/analyze AAPL 10000
/analyze MSFT 5000
/analyze GOOGL 7500

Далее я загружу и проанализирую данные за 2 года, это займет несколько минут, и построю прогноз на 30 дней вперед.
В ответ я вам вышлю картинку прогноза стоимости акций и рекомендации.
Для остановки бота введите /exit или /stop.
Для просмотра доступных команд введите /help
"""
    await message.reply(welcome_msg, parse_mode="Markdown")

@dp.message(Command("help"))
async def command_help_handler(message: types.Message):
    help_text = """
*Справка по использованию бота:*

*Доступные команды:*
/start - Начало работы
/help - Помощь
/analyze \[ТИКЕР] \[СУММА] - Анализ акций
/exit и /stop - Остановка работы

*Тикеры компаний:*
   - Apple: AAPL
   - Microsoft: MSFT
   - NVIDIA: NVDA
   - Google: GOOGL
   - Amazon: AMZN
   - Tesla: TSLA
   - И другие популярные тикеры

*Где найти тикеры:*
Полный список тикеров вы можете найти на сайте https://finance.yahoo.com/

*Что вы получите:*
- График прогноза на 30 дней
- Оценку изменения цены
- Рекомендации по покупке/продаже
- Расчёт потенциальной прибыли
"""
    await message.answer(help_text, parse_mode="Markdown")

@dp.message(Command("analyze"))
async def command_analyze_handler(message: types.Message):
    # Получаем аргументы команды
    args = message.text.split()

    # Проверяем количество аргументов
    if len(args) != 3:
        await message.answer(
            "❌ *Неверный формат команды!*\n\n"
            "*Правильный формат:*\n"
            "```\n/analyze \[ТИКЕР] \[СУММА]\n```\n"
            "*Примеры:*\n"
            "```\n"
            "/analyze AAPL 10000\n"
            "/analyze MSFT 5000\n"
            "```\n"
            "Используйте /help для подробной справки.",
            parse_mode="Markdown"
        )
        return
    
    # Извлекаем тикер и сумму инвестиции
    ticker = args[1].upper().strip()
    amount_str = args[2].strip()

    # Проверяем тикер
    if not re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,3})?$', ticker):
        await message.answer(
            f"❌ *Неверный формат тикера:* {ticker}\n"
            "Тикер должен состоять из 1-5 заглавных букв латинского алфавита.\n"
            "*Примеры:* AAPL, MSFT, GOOGL",
            parse_mode="Markdown"
        )
        return
    
    # Проверяем сумму
    try:
        money = float(amount_str)
        if money <= 0:
            await message.answer("❌ Сумма должна быть положительной.")
            return
        if money > 1000000:  # Ограничение на максимальную сумму
            await message.answer("❌ Сумма не должна превышать $1,000,000.")
            return
    except ValueError:
        await message.answer("❌ Сумма должна быть числом (например: 10000 или 5000.50)")
        return
    
    # Проверяем тикер на существование
    try:
        await message.answer(f"🔍 Проверяю тикер {ticker}...")
        
        end_date = datetime.now()
        start_date = datetime(end_date.year, end_date.month, end_date.day - 5)
        test_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if test_data.empty:
            await message.answer(
                f"❌ Тикер '{ticker}' не найден или нет данных.\n"
                "Проверьте правильность написания тикера.\n"
                "Используйте /help для списка популярных тикеров."
            )
            return
        
        # Информируем пользователя о начале анализа
        await message.answer(f"""
📊 *Начинаю анализ:*
- Тикер: {ticker}
- Сумма: ${money:,.2f}
- Период: 2 года
- Прогноз: 30 дней

⏳ Загружаю данные и обучаю модели...
Это займет 1-2 минуты.
""", parse_mode="Markdown")
        
        # Запускаем анализ в фоновом режиме
        asyncio.create_task(perform_analysis(message, ticker, money, message.from_user.id))
        
    except Exception as e:
        await message.answer(f"❌ Ошибка при проверке тикера: {str(e)}\nПопробуйте другой тикер.")

@dp.message(Command("stop", "exit"))
async def command_stop_handler(message: types.Message):
    await message.answer("Приятно было поработать с Вами! До новых встреч! 👋")
    # dp.stop_polling()

@dp.message(F.text)
async def handle_other_messages(message: types.Message):
    """Обработка других текстовых сообщений"""
    text = message.text.strip()
    
    # Если сообщение похоже на команду /analyze без слеша
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,3})?\s+\d+', text.upper()):
        parts = text.split()
        if len(parts) == 2:
            await message.answer(
                f"🤖 Кажется, вы хотите проанализировать {parts[0].upper()}.\n"
                f"Используйте формат команды:\n"
                f"```\n/analyze {parts[0].upper()} {parts[1]}\n```",
                parse_mode="Markdown"
            )
            return
    
    # Если введен только тикер
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,3})?$', text.upper()):
        await message.answer(
            f"📊 Для анализа акций {text.upper()} введите:\n"
            f"```\n/analyze {text.upper()} \[СУММА]\n```\n"
            f"*Пример:* /analyze {text.upper()} 10000",
            parse_mode="Markdown"
        )
        return
    
    # Общий ответ для непонятных сообщений
    await message.answer("""
🤖 Я не понимаю эту команду.
Используйте /help для получения справки по работе с ботом.
""", parse_mode="Markdown")    

async def perform_analysis(message: types.Message, ticker: str, money: float, user_id: int):
    """Выполнение анализа в фоновом режиме"""
    try:
        # Загрузка данных
        await message.answer("📥 Загружаю исторические данные...")
        end_date = datetime.now()
        start_date = datetime(end_date.year - 2, end_date.month, end_date.day)
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            timeout=10
        )

        if data.empty:
            await message.answer(f"❌ Не удалось загрузить данные для {ticker}")
            return
     
        if len(data) < 60:  # Минимум 60 дней данных
            await message.answer(f"❌ Недостаточно данных для анализа {ticker}")
            return
     
        # Инициализация прогнозировщика
        forecaster = StockForecaster(data['Close'])
     
        # Обучение моделей
        await message.answer("🤖 Обучаю модели...\n1. Random Forest\n2. ARIMA\n3. LSTM")
        best_model_name, best_metric, forecast = forecaster.train_and_forecast()
     
        # Построение графика
        await message.answer("📈 Строю график прогноза...")
        fig = forecaster.plot_forecast(forecast)
     
        # Сохранение графика в буфер
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
     
        # Анализ для инвестиций
        await message.answer("💡 Анализирую торговые возможности...")
        recommendations = forecaster.get_investment_recommendations(forecast, money)

        recommendations_comments = ""
        if recommendations["potential_profit"] == 0 and recommendations["price_change_percent"] < 0:
            recommendations_comments = "⚠️ Не рекомендуется покупать. Отложите покупку до улучшения ситуации"
        else:
            recommendations_comments = recommendations["summary"]

        # Формирование отчета
        report = f"""
*📊 ОТЧЕТ ПО АНАЛИЗУ АКЦИЙ {ticker}*

*📈 Прогноз на 30 дней:*
- Лучшая модель: *{best_model_name}*
- Метрика RMSE: *{best_metric:.4f}*
- Прогноз изменения: *{recommendations["price_change_percent"]:+.2f}%*

*💰 Инвестиционные рекомендации:*
- Сумма инвестиций: *${money:,.2f}*
- Потенциальная прибыль: *${recommendations["potential_profit"]:,.2f}*
- ROI (Return on Investment): *{recommendations["roi"]:+.2f}%*
- Итоговая стоимость: *${(recommendations["potential_profit"]+ money):,.2f}*

*📅 Рекомендуемые действия:*
{recommendations_comments}

*📊 Статистика прогноза:*
- Минимальная цена: ${forecast.min():.2f}
- Максимальная цена: ${forecast.max():.2f}
- Средняя цена: ${forecast.mean():.2f}
"""
     
        # Отправка графика и отчета
        await message.answer_photo(
            types.BufferedInputFile(buf.read(), filename="forecast.png"),
            caption=report,
            parse_mode="Markdown"
        )
     
        # Сохранение сессии в логи
        log_data = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'money': money,
            'best_model': best_model_name,
            'rmse': best_metric,
            'potential_profit': recommendations['potential_profit'],
            'roi': recommendations['roi'],
            'price_change': recommendations['price_change_percent']
        }
        save_to_log(log_data)
     
        await message.answer("✅ Анализ завершен! Данные сохранены в лог.")
     
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка при анализе: {str(e)}")
        logging.error(f"Error in analysis: {e}", exc_info=True)

async def main():
    logging.basicConfig(level=logging.INFO) # Включаем логирование
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())