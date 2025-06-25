import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

def prepare_data_for_prediction(data: pd.DataFrame) -> tuple:
    """
    تحضير البيانات للتنبؤ
    
    :param data: بيانات السهم التاريخية
    :return: (الميزات، الهدف)
    """
    try:
        # حساب المتوسط المتحرك
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        
        # حساب RSI (بديل إذا لم يكن TA-Lib مثبتاً)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # إزالة القيم NaN
        data = data.dropna()
        
        # تحضير الميزات والهدف
        features = data[['SMA_20', 'RSI']]
        target = data['close'].shift(-1).dropna()
        features = features.iloc[:-1]
        
        return features, target
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_prediction_model(features: pd.DataFrame, target: pd.Series) -> tuple:
    """
    تدريب نموذج التنبؤ
    
    :param features: بيانات الميزات
    :param target: بيانات الهدف
    :return: (النموذج، خطأ MSE)
    """
    try:
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # تدريب النموذج
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # تقييم النموذج
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        return model, mse
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def predict_next_day(model, last_data_point: pd.Series) -> float:
    """
    التنبؤ بسعر اليوم التالي
    
    :param model: النموذج المدرب
    :param last_data_point: آخر نقطة بيانات
    :return: السعر المتوقع
    """
    try:
        # تحضير بيانات التنبؤ
        pred_features = pd.DataFrame({
            'SMA_20': [last_data_point['SMA_20']],
            'RSI': [last_data_point['RSI']]
        })
        
        # التنبؤ
        prediction = model.predict(pred_features)
        return prediction[0]
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise
