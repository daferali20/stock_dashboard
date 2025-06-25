import yfinance as yf
import pandas as pd
from typing import Optional

def get_analyst_recommendations(ticker: str) -> Optional[pd.DataFrame]:
    """
    جلب توصيات المحللين لسهم معين
    
    :param ticker: رمز السهم
    :return: DataFrame يحتوي على التوصيات أو None إذا فشل
    """
    try:
        stock = yf.Ticker(ticker)
        rec = stock.recommendations
        
        if rec is None or rec.empty:
            return None
            
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
