import pandas as pd
import talib
from typing import Optional, Tuple
from utils.helpers import validate_data

class TechnicalIndicators:
    """
    حساب المؤشرات الفنية للأسهم باستخدام TA-Lib
    """
    
    def __init__(self, ohlc_data: pd.DataFrame):
        """
        تهيئة الكائن ببيانات OHLC (Open, High, Low, Close)
        
        :param ohlc_data: DataFrame يحتوي على الأعمدة:
            ['open', 'high', 'low', 'close', 'volume'] (اختياري)
        """
        validate_data(ohlc_data)
        self.data = ohlc_data.copy()
        self.close = self.data['close'].values
        self.high = self.data['high'].values
        self.low = self.data['low'].values
        self.open = self.data['open'].values
        self.volume = self.data.get('volume', None)

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        حساب مؤشر القوة النسبية (RSI)
        
        :param period: الفترة الزمنية (افتراضي 14)
        :return: سلسلة pandas تحتوي على قيم RSI
        """
        rsi = talib.RSI(self.close, timeperiod=period)
        return pd.Series(rsi, index=self.data.index, name='RSI')

    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, 
                      signal_period: int = 9) -> pd.DataFrame:
        """
        حساب مؤشر MACD (Moving Average Convergence Divergence)
        
        :return: DataFrame يحتوي على:
            - MACD Line
            - Signal Line
            - MACD Histogram
        """
        macd, signal, hist = talib.MACD(
            self.close, 
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': hist
        }, index=self.data.index)

    def calculate_bollinger_bands(self, period: int = 20, 
                                nbdev: float = 2.0) -> pd.DataFrame:
        """
        حساب عصابات بولينجر
        
        :return: DataFrame يحتوي على:
            - Upper Band
            - Middle Band (SMA)
            - Lower Band
        """
        upper, middle, lower = talib.BBANDS(
            self.close,
            timeperiod=period,
            nbdevup=nbdev,
            nbdevdn=nbdev
        )
        return pd.DataFrame({
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        }, index=self.data.index)
    # بدائل إذا لم يكن TA-Lib مثبتاً
    def calculate_rsi(self, period=14):
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_sma(self, period: int = 20) -> pd.Series:
        """المتوسط المتحرك البسيط"""
        sma = talib.SMA(self.close, timeperiod=period)
        return pd.Series(sma, index=self.data.index, name=f'SMA_{period}')

    def calculate_ema(self, period: int = 20) -> pd.Series:
        """المتوسط المتحرك الأسي"""
        ema = talib.EMA(self.close, timeperiod=period)
        return pd.Series(ema, index=self.data.index, name=f'EMA_{period}')

    def calculate_stochastic(self, k_period: int = 14, 
                           d_period: int = 3) -> pd.DataFrame:
        """
        حساب مؤشر ستوكاستيك
        
        :return: DataFrame يحتوي على:
            - %K Line
            - %D Line (المتوسط المتحرك لـ %K)
        """
        slowk, slowd = talib.STOCH(
            self.high,
            self.low,
            self.close,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        return pd.DataFrame({
            '%K': slowk,
            '%D': slowd
        }, index=self.data.index)

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """متوسط المدى الحقيقي (ATR)"""
        atr = talib.ATR(self.high, self.low, self.close, timeperiod=period)
        return pd.Series(atr, index=self.data.index, name='ATR')

    def calculate_obv(self) -> pd.Series:
        """حجم الرصيد (OBV) - يتطلب عمود 'volume'"""
        if self.volume is None:
            raise ValueError("بيانات الحجم مطلوبة لحساب OBV")
        obv = talib.OBV(self.close, self.volume)
        return pd.Series(obv, index=self.data.index, name='OBV')

    def calculate_all_indicators(self) -> pd.DataFrame:
        """حساب جميع المؤشرات وإرجاعها في DataFrame واحد"""
        indicators = self.data.copy()
        
        # إضافة كل المؤشرات
        indicators['RSI'] = self.calculate_rsi()
        indicators[['MACD', 'Signal', 'Histogram']] = self.calculate_macd()
        indicators[['BB_Upper', 'BB_Middle', 'BB_Lower']] = self.calculate_bollinger_bands()
        indicators['SMA_20'] = self.calculate_sma()
        indicators['EMA_20'] = self.calculate_ema()
        indicators[['Stoch_%K', 'Stoch_%D']] = self.calculate_stochastic()
        indicators['ATR'] = self.calculate_atr()
        
        try:
            indicators['OBV'] = self.calculate_obv()
        except ValueError:
            pass
            
        return indicators
