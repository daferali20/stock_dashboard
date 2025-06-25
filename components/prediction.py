import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

def predict_stock(data: pd.DataFrame, days: int = 1) -> dict:
    """
    تنبؤ بسيط بحركة السهم باستخدام الانحدار الخطي
    
    :param data: بيانات السهم مع المؤشرات الفنية
    :param days: عدد الأيام للتنبؤ
    :return: قاموس يحتوي على:
        - direction (صعود/هبوط)
        - confidence (نسبة الثقة)
        - price (السعر المتوقع)
    """
    try:
        # تحضير البيانات
        data = data.copy().dropna()
        X = data[['SMA_20', 'RSI', 'MACD']].values[:-days]
        y = data['close'].shift(-days).dropna().values
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # تدريب النموذج
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # التنبؤ
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        last_data = data[['SMA_20', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1)
        next_price = model.predict(last_data)[0]
        
        return {
            'direction': 'صعود' if next_price > data['close'].iloc[-1] else 'هبوط',
            'confidence': max(0, min(1, 1 - (mse / data['close'].mean()))),
            'price': next_price
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return {
            'direction': 'غير متاح',
            'confidence': 0,
            'price': data['close'].iloc[-1] if len(data) > 0 else 0
        }
