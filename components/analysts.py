import yfinance as yf
import pandas as pd
from typing import Optional

def get_analyst_recommendations(ticker: str) -> pd.DataFrame:
    """جلب توصيات المحللين من Alpha Vantage"""
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "AnalystTargetPrice" in data:
            return pd.DataFrame({
                "Target Price": [data["AnalystTargetPrice"]],
                "Rating": [data["AnalystRating"]],
                "Firm": [data["AnalystFirm"]]
            })
            
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return pd.DataFrame()
            
        # توحيد أسماء الأعمدة
        rec.columns = rec.columns.str.lower()
        
        # اختيار الأعمدة المطلوبة
        required_cols = ['firm', 'to grade', 'action']
        available_cols = [col for col in required_cols if col in rec.columns]
        
        if not available_cols:
            return None
            
        return rec[available_cols].sort_values(by='to grade', ascending=False)
        
    except Exception as e:
        print(f"Error getting recommendations for {ticker}: {str(e)}")
        return None
