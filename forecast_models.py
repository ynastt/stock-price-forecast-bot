import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import argrelextrema
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    """LSTM модель для прогнозирования временных рядов"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    """Датасет для временных рядов"""
    def __init__(self, data, sequence_length=30):
        self.data = data
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length]
        y = self.data[idx+self.sequence_length]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y])

class StockForecaster:
    """Класс для прогнозирования акций"""
    
    def __init__(self, price_series):
        """
        Инициализация прогнозировщика
        
        Args:
            price_series: pandas Series с ценами закрытия
        """
        self.prices = price_series
        self.data = price_series.values.reshape(-1, 1)
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        
    def prepare_features(self, data, n_lags=30):
        """Подготовка признаков для ML моделей"""
        X, y = [], []
        for i in range(n_lags, len(data)):
            X.append(data[i-n_lags:i].flatten())
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_random_forest(self, test_size=0.2):
        """Обучение Random Forest модели"""
        # Подготовка данных
        X, y = self.prepare_features(self.scaled_data)
        
        # Разделение на train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Обучение модели
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Прогноз
        y_pred = model.predict(X_test)
        
        # Обратное масштабирование
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Расчет метрик
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
        
        return model, rmse, mape
    
    def train_arima(self, test_size=0.2):
        """Обучение ARIMA модели"""
        # Разделение на train/test
        split_idx = int(len(self.prices) * (1 - test_size))
        train_series = self.prices.iloc[:split_idx]
        test_series = self.prices.iloc[split_idx:]
        
        try:
            # Пробуем ARIMA(1,1,1) - стандартные параметры
            model = ARIMA(train_series, order=(1,1,1))
            fitted_model = model.fit()
            
            # Прогноз на тестовых данных
            forecast = fitted_model.forecast(steps=len(test_series))
            
            # Расчет метрик
            rmse = np.sqrt(mean_squared_error(test_series.values, forecast.values))
            mape = mean_absolute_percentage_error(test_series.values, forecast.values)
            
            return fitted_model, rmse, mape
            
        except Exception as e:
            print(f"Ошибка ARIMA: {e}")
            # Создаем простую модель как запасной вариант
            class SimpleARIMA:
                def __init__(self):
                    self.last_value = None
                
                def fit(self, data):
                    self.last_value = data.iloc[-1]
                    return self
                
                def forecast(self, steps):
                    return pd.Series([self.last_value] * steps)
            
            simple_model = SimpleARIMA()
            simple_model.fit(train_series)
            forecast = simple_model.forecast(len(test_series))
            
            rmse = np.sqrt(mean_squared_error(test_series.values, forecast.values))
            mape = mean_absolute_percentage_error(test_series.values, forecast.values)
            
        return simple_model, rmse, mape
    

    def train_lstm(self, test_size=0.2, epochs=50):
        """Обучение LSTM модели"""
        # Подготовка данных
        dataset = TimeSeriesDataset(self.scaled_data.flatten())
        split_idx = int(len(dataset) * (1 - test_size))
        
        train_data = torch.utils.data.Subset(dataset, range(split_idx))
        test_data = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        # Инициализация модели
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Обучение
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Прогноз на тестовых данных
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                predictions.extend(outputs.numpy().flatten())
                actuals.extend(batch_y.numpy().flatten())
        
        # Обратное масштабирование
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        actuals = self.scaler.inverse_transform(
            np.array(actuals).reshape(-1, 1)
        ).flatten()
        
        # Расчет метрик
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = mean_absolute_percentage_error(actuals, predictions)
        
        return model, rmse, mape
    
    def train_and_forecast(self):
        """Обучение всех моделей и выбор лучшей"""
        print("Обучение моделей...")
        
        # Обучение Random Forest
        print("1. Обучение Random Forest...")
        rf_model, rf_rmse, rf_mape = self.train_random_forest()
        
        # Обучение ARIMA
        print("2. Обучение ARIMA...")
        arima_model, arima_rmse, arima_mape = self.train_arima()
        
        # Обучение LSTM
        print("3. Обучение LSTM...")
        lstm_model, lstm_rmse, lstm_mape = self.train_lstm(epochs=30)
        
        # Выбор лучшей модели по RMSE
        models = {
            'Random Forest': (rf_model, rf_rmse),
            'ARIMA': (arima_model, arima_rmse),
            'LSTM': (lstm_model, lstm_rmse)
        }
        
        print("\n" + "="*50)
        print("Сравнение моделей:")
        print("-"*50)
        for name, (_, rmse) in models.items():
            print(f"{name:15} | RMSE = {rmse:8.4f}")
        print("="*50)

        best_model_name = min(models, key=lambda x: models[x][1])
        best_model, best_rmse = models[best_model_name]
        
        print(f"Лучшая модель: {best_model_name} (RMSE: {best_rmse:.4f})")
        
        # Построение прогноза лучшей моделью
        forecast = self.make_forecast(best_model_name, best_model)
        
        return best_model_name, best_rmse, forecast
    
    def make_forecast(self, model_name, model):
        """Построение прогноза на 30 дней"""
        if model_name == 'Random Forest':
            # Для Random Forest используем последние 30 значений
            last_values = self.scaled_data[-30:].flatten()
            forecast = []
            
            for _ in range(30):
                X_input = last_values[-30:].reshape(1, -1)
                pred = model.predict(X_input)[0]
                forecast.append(pred)
                last_values = np.append(last_values, pred)
            
            forecast = self.scaler.inverse_transform(
                np.array(forecast).reshape(-1, 1)
            ).flatten()
            
        elif model_name == 'ARIMA':
            try:
                # Если это statsmodels ARIMA
                if hasattr(model, 'forecast'):
                    forecast = model.forecast(steps=30).values
                elif hasattr(model, 'predict'):
                    # Для простой модели
                    forecast = model.forecast(30).values if hasattr(model, 'forecast') else model.predict(30)
                else:
                    # Запасной вариант
                    forecast = np.full(30, self.prices.iloc[-1])
            except Exception as e:
                print(f"Ошибка прогноза ARIMA: {e}")
                forecast = np.full(30, self.prices.iloc[-1])
            
        else:  # LSTM
            model.eval()
            forecast = []

            # Подготавливаем начальную последовательность
            # Размерность должна быть: (batch_size, sequence_length, input_size)
            if len(self.scaled_data) >= 30:
                last_sequence = torch.FloatTensor(
                    self.scaled_data[-30:].flatten()
                ).unsqueeze(-1).unsqueeze(0)  # размерность: (1, 30, 1)
            else:
                # Если данных меньше 30, дополняем
                padded_data = np.pad(
                    self.scaled_data.flatten(), 
                    (30 - len(self.scaled_data), 0), 
                    'constant', 
                    constant_values=np.mean(self.scaled_data)
                )
                last_sequence = torch.FloatTensor(padded_data).unsqueeze(-1).unsqueeze(0)
            
            with torch.no_grad():
                for _ in range(30):
                    pred = model(last_sequence)
                    forecast.append(pred.item())
                    # Обновляем последовательность
                    # 1. Убираем первый элемент последовательности
                    # 2. Добавляем новое предсказание в конец
                
                    # Преобразуем pred к правильной размерности: (1, 1, 1)
                    pred_reshaped = pred.unsqueeze(-1)  # теперь размерность: (1, 1, 1)
                
                    # Объединяем: [последние 29 элементов] + [новое предсказание]
                    last_sequence = torch.cat([
                        last_sequence[:, 1:, :],  # размерность: (1, 29, 1)
                        pred_reshaped            # размерность: (1, 1, 1)
                    ], dim=1)  # результат: (1, 30, 1)
            
            forecast = self.scaler.inverse_transform(
                np.array(forecast).reshape(-1, 1)
            ).flatten()
        
        return forecast
    
    def plot_forecast(self, forecast):
        """Построение графика с прогнозом"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Исторические данные
        ax.plot(self.prices.index, self.prices.values, 
                label='Исторические данные', linewidth=2, color='blue')
        
        # Прогноз
        forecast_dates = pd.date_range(
            start=self.prices.index[-1] + timedelta(days=1),
            periods=len(forecast),
            freq='D'
        )
        ax.plot(forecast_dates, forecast, 
                label='Прогноз на 30 дней', linewidth=2, color='red', linestyle='--')
        
        # Настройки графика
        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Цена ($)', fontsize=12)
        ax.set_title('Прогноз цен акций', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси X
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def get_investment_recommendations(self, forecast, investment_amount):
        """Генерация инвестиционных рекомендаций"""
        if len(forecast) < 5:
            return {
                'potential_profit': 0,
                'roi': 0,
                'trades': [],
                'summary': "Недостаточно данных для анализа прогноза",
                'price_change_percent': 0
            }
        
        # Находим локальные минимумы и максимумы
        # Локальные минимумы (покупка)
        # Локальные максимумы (продажа)
        try:
            # Пробуем разные параметры для поиска экстремумов
            # order=1 для более чувствительного поиска
            min_indices = argrelextrema(forecast, np.less, order=1)[0]
            max_indices = argrelextrema(forecast, np.greater, order=1)[0]
        except:
            # Если не удалось найти экстремумы
            min_indices = []
            max_indices = []

        print(len(min_indices), len(max_indices))

        # Если нет экстремумов, используем простую стратегию
        if len(min_indices) == 0 or len(max_indices) == 0:
            return self.simple_investment_strategy(forecast, investment_amount)
        
        # Симуляция торговли
        cash = investment_amount
        shares = 0
        trades = []
        
        # стратегия: покупаем на минимумах, продаем на максимумах
        # Собираем все точки сделок
        trade_points = []
        for idx in min_indices:
            if 0 <= idx < len(forecast):
                trade_points.append(('BUY', idx, forecast[idx]))
        for idx in max_indices:
            if 0 <= idx < len(forecast):
                trade_points.append(('SELL', idx, forecast[idx]))
        
        # Сортируем по времени
        trade_points.sort(key=lambda x: x[1])

        for action, idx, price in trade_points:
            date = pd.date_range(
                start=self.prices.index[-1] + timedelta(days=1),
                periods=len(forecast),
                freq='D'
            )[idx]
        
            if action == 'BUY' and cash > 0 and idx < len(forecast) - 1:
                # Покупаем часть доступных средств
                buy_amount = cash * 0.8
                shares_bought = buy_amount / price
                shares += shares_bought
                cash -= buy_amount
            
                trades.append({
                    'date': date,
                    'action': 'Покупка',
                    'price': price,
                    'shares': shares_bought,
                    'amount': buy_amount,
                    'cash_after': cash
                })
            
            elif action == 'SELL' and shares > 0 and idx > 0:
                # Продаем часть акций
                shares_to_sell = shares * 0.8
                sell_amount = shares_to_sell * price
                cash += sell_amount
                shares -= shares_to_sell
            
                trades.append({
                    'date': date,
                    'action': 'Продажа',
                    'price': price,
                    'shares': shares_to_sell,
                    'amount': sell_amount,
                    'cash_after': cash
                })
        # all_indices = sorted(set(min_indices) | set(max_indices))
        
        # for idx in all_indices:
        #     price = forecast[idx]
        #     date = pd.date_range(
        #         start=self.prices.index[-1] + timedelta(days=1),
        #         periods=len(forecast),
        #         freq='D'
        #     )[idx]
            
        #     if idx in min_indices and cash > 0:
        #         # Покупаем
        #         shares_to_buy = cash / price
        #         shares += shares_to_buy
        #         trades.append({
        #             'date': date,
        #             'action': 'Покупка',
        #             'price': price,
        #             'shares': shares_to_buy,
        #             'cash_before': cash,
        #             'cash_after': 0
        #         })
        #         cash = 0
                
        #     elif idx in max_indices and shares > 0:
        #         # Продаем
        #         cash = shares * price
        #         trades.append({
        #             'date': date,
        #             'action': 'Продажа',
        #             'price': price,
        #             'shares': shares,
        #             'cash_before': 0,
        #             'cash_after': cash
        #         })
        #         shares = 0
        
        # Финализируем позицию (продаем в конце если есть акции)
        if shares > 0:
            final_price = forecast[-1]
            cash = shares * final_price
            trades.append({
                'date': pd.date_range(
                    start=self.prices.index[-1] + timedelta(days=1),
                    periods=len(forecast),
                    freq='D'
                )[-1],
                'action': 'Продажа',
                'price': final_price,
                'shares': shares,
                'cash_before': 0,
                'cash_after': cash
            })
        
        # Расчет прибыли
        final_value = cash
        profit = final_value - investment_amount
        roi = (profit / investment_amount) * 100
        
        # Формирование сводки
        if trades:
            summary = []
            for trade in trades:
                summary.append(
                    f"{trade['date'].strftime('%d.%m')}: {trade['action']} по ${trade['price']:.2f}"
                )
            print("summary:")
            print(summary)
        else:
            summary = ["📊 Нет торговых сигналов в прогнозе"]
        
        # Изменение цены
        price_change = forecast[-1] - forecast[0]
        price_change_percent = (price_change / forecast[0]) * 100 if forecast[0] != 0 else 0
        
        return {
            'potential_profit': max(profit, 0),
            'roi': roi,
            'trades': trades,
            'summary': "\n".join(summary),
            'price_change_percent': price_change_percent,
            'final_value': final_value
        }
    
    def simple_investment_strategy(self, forecast, investment_amount):
        """Простая стратегия для случаев без экстремумов"""
        if len(forecast) < 2:
            return {
                'potential_profit': 0,
                'roi': 0,
                'trades': [],
                'summary': "Недостаточно данных для анализа",
                'price_change_percent': 0,
                'final_value': investment_amount
            }   
    
        # Простая стратегия: покупаем в начале, продаем в конце
        buy_price = forecast[0]
        sell_price = forecast[-1]
    
        shares = investment_amount / buy_price
        final_value = shares * sell_price
        profit = final_value - investment_amount
        roi = (profit / investment_amount) * 100
    
        price_change = sell_price - buy_price
        price_change_percent = (price_change / buy_price) * 100 if buy_price != 0 else 0
    
        summary = f"""
📊 *Простая стратегия:*

• Покупка (день 1): ${buy_price:.2f}
• Продажа (день {len(forecast)}): ${sell_price:.2f}
• Изменение цены: {price_change_percent:+.2f}%
• Прибыль: ${profit:+,.2f} ({roi:+.2f}%)
"""
    
        return {
            'potential_profit': max(profit, 0),
            'roi': roi,
            'trades': [],
            'summary': summary,
            'price_change_percent': price_change_percent,
            'final_value': final_value
        }