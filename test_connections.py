#!/usr/bin/env python3
"""
Скрипт для тестирования соединения с Yahoo Finance
"""
import sys
import os
from datetime import datetime

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.ssl_fix import fix_ssl_issues
import yfinance as yf
import pandas as pd

def test_yfinance():
    """Тестирование соединения с Yahoo Finance"""
    print("Тестирование соединения с Yahoo Finance...")
    
    # Тестовые тикеры
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\nПробую загрузить {ticker}...")
        try:
            # Пробуем загрузить 1 день данных
            # data = yf.download(ticker, period='1d', progress=False)
            # Загружаем данные через yfinance
            stock = yf.Ticker(ticker)
            end = datetime.now()
            start = datetime(end.year, end.month, end.day - 1)
            print(f"start date: {start}, end date: {end}\n")
            data = stock.history(start=start, end=end)
            if not data.empty:
                print(f"✅ {ticker}: Успешно! Цена закрытия: {data['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ {ticker}: Нет данных")
        except Exception as e:
            print(f"❌ {ticker}: Ошибка: {e}")
    
    print("\n" + "="*50)
    print("Если есть ошибки SSL, попробуйте:")
    print("1. pip install --upgrade certifi")
    print("2. pip install --upgrade yfinance")
    print("3. Установите корневые сертификаты Windows")
    print("4. Или используйте VPN")

if __name__ == "__main__":
    # Применяем исправления SSL
    fix_ssl_issues()
    
    # Тестируем
    test_yfinance()