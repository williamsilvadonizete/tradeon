import pandas as pd
import numpy as np
import ta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import sys
import os

# Adiciona o diretório raiz ao path para importar configurações
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    RSI_PERIOD, EMA_FAST, EMA_SLOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD
)

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, df):
        """
        Calcula todos os indicadores técnicos para um DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preço
            
        Returns:
            pd.DataFrame: DataFrame com indicadores adicionados
        """
        if df is None or df.empty:
            return None
            
        # RSI
        rsi = RSIIndicator(close=df['close'], window=RSI_PERIOD)
        df['rsi'] = rsi.rsi()
        
        # EMAs
        ema_fast = EMAIndicator(close=df['close'], window=EMA_FAST)
        ema_slow = EMAIndicator(close=df['close'], window=EMA_SLOW)
        df['ema_fast'] = ema_fast.ema_indicator()
        df['ema_slow'] = ema_slow.ema_indicator()
        
        # MACD
        macd = MACD(
            close=df['close'],
            window_fast=MACD_FAST,
            window_slow=MACD_SLOW,
            window_sign=MACD_SIGNAL
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(
            close=df['close'],
            window=BB_PERIOD,
            window_dev=BB_STD
        )
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        
        return df
        
    def get_signal_strength(self, df):
        """
        Calcula a força do sinal baseado em múltiplos indicadores.
        
        Args:
            df (pd.DataFrame): DataFrame com indicadores
            
        Returns:
            float: Força do sinal (-1 a 1)
        """
        if df is None or df.empty:
            return 0
            
        # Obtém o último valor de cada indicador
        last = df.iloc[-1]
        
        # RSI
        rsi_signal = 0
        if last['rsi'] < 30:
            rsi_signal = 1  # Sobre-vendido
        elif last['rsi'] > 70:
            rsi_signal = -1  # Sobre-comprado
            
        # EMA
        ema_signal = 1 if last['ema_fast'] > last['ema_slow'] else -1
        
        # MACD
        macd_signal = 1 if last['macd'] > last['macd_signal'] else -1
        
        # Bollinger Bands
        bb_signal = 0
        if last['close'] < last['bb_low']:
            bb_signal = 1  # Preço abaixo da banda inferior
        elif last['close'] > last['bb_high']:
            bb_signal = -1  # Preço acima da banda superior
            
        # Combina os sinais
        total_signal = (rsi_signal + ema_signal + macd_signal + bb_signal) / 4
        
        return total_signal
        
    def get_trend_strength(self, df):
        """
        Calcula a força da tendência atual.
        
        Args:
            df (pd.DataFrame): DataFrame com indicadores
            
        Returns:
            float: Força da tendência (0 a 1)
        """
        if df is None or df.empty:
            return 0
            
        # Calcula a direção da tendência
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        # Calcula a consistência da tendência
        ema_trend = (df['ema_fast'].iloc[-1] - df['ema_fast'].iloc[-20]) / df['ema_fast'].iloc[-20]
        
        # Combina os fatores
        trend_strength = abs(price_change * 0.7 + ema_trend * 0.3)
        
        return min(trend_strength, 1.0)  # Limita a 1.0 