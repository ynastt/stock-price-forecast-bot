import pandas as pd
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()
LOG_FILE = os.getenv('LOG_FILE')

if not LOG_FILE:
    raise ValueError("LOG_FILE не найден в переменных окружения. Создайте файл .env")

def save_to_log(log_data: dict):
    """Сохранение данных в лог-файл"""
    log_file = LOG_FILE
    
    # Создаем DataFrame из данных
    df_new = pd.DataFrame([log_data])
    
    # Если файл существует, загружаем и добавляем новые данные
    if os.path.exists(log_file):
        df_existing = pd.read_csv(log_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Сохраняем в CSV
    df_combined.to_csv(log_file, index=False)
    
    print(f"Данные сохранены в лог: {log_file}")
