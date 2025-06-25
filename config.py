# config settings
import requests
import pandas as pd
from config import ALPHA_VANTAGE_API_KEY

BASE_URL = "https://www.alphavantage.co/query"

def get_stock_data(symbol: str, interval="daily") -> pd.DataFrame:
    """جلب بيانات الأسهم من Alpha Vantage"""
    params = {
        "function": "TIME_SERIES_DAILY" if interval == "daily" else "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact",
        "datatype": "json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(
                data["Time Series (Daily)"], 
                orient="index"
            )
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            })
            df.index = pd.to_datetime(df.index)
            return df.astype(float).sort_index()
            
        raise ValueError(f"لا توجد بيانات متاحة للسهم {symbol}")
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.DataFrame()
