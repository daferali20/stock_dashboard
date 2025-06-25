import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st

@st.cache_data(ttl=3600)
def prepare_data_for_prediction(data: pd.DataFrame):
    try:
        if data.empty or 'Close' not in data.columns:
            return pd.DataFrame(), pd.Series()

        data['Price_Up'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=5).std()
        data = data.dropna()

        features = data[['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'Volatility']]
        target = data['Price_Up']

        return features, target
    except Exception as e:
        st.error(f"خطأ أثناء تحضير بيانات التنبؤ: {str(e)}")
        return pd.DataFrame(), pd.Series()

def train_prediction_model(features: pd.DataFrame, target: pd.Series):
    try:
        if features.empty or target.empty or len(features) < 30:
            return None, 0

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=False
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        return model, mse
    except Exception as e:
        st.error(f"خطأ أثناء تدريب النموذج: {str(e)}")
        return None, 0

def predict_next_day(model, last_data: pd.Series):
    try:
        if model is None or last_data.empty:
            return 0.5

        required_cols = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'Volatility']
        if not all(col in last_data.index for col in required_cols):
            return 0.5

        last_features = last_data[required_cols].values.reshape(1, -1)
        prediction = model.predict(last_features)[0]
        return max(0, min(1, prediction))
    except:
        return 0.5

