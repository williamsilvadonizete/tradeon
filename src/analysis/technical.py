import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import ta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

@dataclass
class MarketStructure:
    support_levels: List[float]
    resistance_levels: List[float]
    trend: str  # 'uptrend', 'downtrend', 'sideways'
    strength: float  # 0 to 1
    key_levels: List[float]

@dataclass
class VolumeProfile:
    price_levels: List[float]
    volume_at_price: List[float]
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float

class AdvancedTechnicalAnalysis:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurações dos indicadores
        self.rsi_period = config.get('rsi_period', 14)
        self.ema_fast = config.get('ema_fast', 9)
        self.ema_slow = config.get('ema_slow', 21)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        self.volume_profile_periods = config.get('volume_profile_periods', 20)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos.
        
        Args:
            data: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com indicadores adicionados
        """
        try:
            # RSI
            rsi = RSIIndicator(close=data['close'], window=self.rsi_period)
            data['rsi'] = rsi.rsi()
            
            # EMAs
            ema_fast = EMAIndicator(close=data['close'], window=self.ema_fast)
            ema_slow = EMAIndicator(close=data['close'], window=self.ema_slow)
            data['ema_fast'] = ema_fast.ema_indicator()
            data['ema_slow'] = ema_slow.ema_indicator()
            
            # MACD
            macd = MACD(close=data['close'])
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = BollingerBands(close=data['close'], window=self.bb_period, window_dev=self.bb_std)
            data['bb_high'] = bb.bollinger_hband()
            data['bb_low'] = bb.bollinger_lband()
            data['bb_mid'] = bb.bollinger_mavg()
            
            # Volatilidade
            data['volatility'] = data['close'].pct_change().rolling(window=20).std()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return data
            
    def analyze_market_structure(self, data: pd.DataFrame) -> MarketStructure:
        """
        Analisa a estrutura do mercado identificando suportes, resistências e tendência.
        
        Args:
            data: DataFrame com dados OHLCV
            
        Returns:
            MarketStructure com níveis e tendência
        """
        try:
            # Identifica pivots
            pivots = self._find_pivot_points(data)
            
            # Identifica níveis de suporte e resistência
            support_levels = self._find_support_levels(data, pivots)
            resistance_levels = self._find_resistance_levels(data, pivots)
            
            # Determina tendência
            trend, strength = self._determine_trend(data)
            
            # Identifica níveis chave
            key_levels = self._find_key_levels(data, support_levels, resistance_levels)
            
            return MarketStructure(
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend=trend,
                strength=strength,
                key_levels=key_levels
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {str(e)}")
            return MarketStructure([], [], 'sideways', 0.0, [])
            
    def calculate_volume_profile(self, data: pd.DataFrame) -> VolumeProfile:
        """
        Calcula o perfil de volume para identificar níveis de preço importantes.
        
        Args:
            data: DataFrame com dados OHLCV
            
        Returns:
            VolumeProfile com níveis de preço e volume
        """
        try:
            # Agrupa preços em níveis
            price_levels = np.linspace(data['low'].min(), data['high'].max(), 100)
            volume_at_price = np.zeros_like(price_levels)
            
            # Calcula volume em cada nível
            for i in range(len(data)):
                for j in range(len(price_levels)-1):
                    if data['low'].iloc[i] <= price_levels[j] <= data['high'].iloc[i]:
                        volume_at_price[j] += data['volume'].iloc[i]
            
            # Encontra Point of Control (POC)
            poc_idx = np.argmax(volume_at_price)
            poc_price = price_levels[poc_idx]
            
            # Calcula Value Area (70% do volume)
            total_volume = np.sum(volume_at_price)
            target_volume = total_volume * 0.7
            
            # Encontra Value Area High e Low
            sorted_volumes = np.sort(volume_at_price)[::-1]
            cumulative_volume = np.cumsum(sorted_volumes)
            value_area_idx = np.where(cumulative_volume >= target_volume)[0][0]
            
            value_area_high = price_levels[np.where(volume_at_price >= sorted_volumes[value_area_idx])[0].max()]
            value_area_low = price_levels[np.where(volume_at_price >= sorted_volumes[value_area_idx])[0].min()]
            
            return VolumeProfile(
                price_levels=price_levels.tolist(),
                volume_at_price=volume_at_price.tolist(),
                poc_price=poc_price,
                value_area_high=value_area_high,
                value_area_low=value_area_low
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {str(e)}")
            return VolumeProfile([], [], 0.0, 0.0, 0.0)
            
    def _find_pivot_points(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """Encontra pontos de pivot no gráfico."""
        pivots = []
        for i in range(2, len(data)-2):
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                pivots.append((i, data['low'].iloc[i]))
            elif (data['high'].iloc[i] > data['high'].iloc[i-1] and
                  data['high'].iloc[i] > data['high'].iloc[i-2] and
                  data['high'].iloc[i] > data['high'].iloc[i+1] and
                  data['high'].iloc[i] > data['high'].iloc[i+2]):
                pivots.append((i, data['high'].iloc[i]))
        return pivots
        
    def _find_support_levels(self, data: pd.DataFrame, pivots: List[Tuple[int, float]]) -> List[float]:
        """Identifica níveis de suporte."""
        support_levels = []
        for i, price in pivots:
            if data['close'].iloc[i] > price:
                support_levels.append(price)
        return sorted(list(set(support_levels)))
        
    def _find_resistance_levels(self, data: pd.DataFrame, pivots: List[Tuple[int, float]]) -> List[float]:
        """Identifica níveis de resistência."""
        resistance_levels = []
        for i, price in pivots:
            if data['close'].iloc[i] < price:
                resistance_levels.append(price)
        return sorted(list(set(resistance_levels)))
        
    def _determine_trend(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Determina a tendência atual e sua força."""
        # Usa EMAs para determinar tendência
        ema_fast = data['ema_fast'].iloc[-1]
        ema_slow = data['ema_slow'].iloc[-1]
        
        # Calcula força da tendência
        strength = abs(ema_fast - ema_slow) / ema_slow
        
        if ema_fast > ema_slow:
            trend = 'uptrend'
        elif ema_fast < ema_slow:
            trend = 'downtrend'
        else:
            trend = 'sideways'
            
        return trend, min(strength, 1.0)
        
    def _find_key_levels(self, data: pd.DataFrame, 
                        support_levels: List[float],
                        resistance_levels: List[float]) -> List[float]:
        """Identifica níveis chave de preço."""
        key_levels = []
        
        # Adiciona níveis de suporte e resistência
        key_levels.extend(support_levels)
        key_levels.extend(resistance_levels)
        
        # Adiciona médias móveis
        key_levels.append(data['ema_fast'].iloc[-1])
        key_levels.append(data['ema_slow'].iloc[-1])
        
        # Adiciona Bollinger Bands
        key_levels.append(data['bb_high'].iloc[-1])
        key_levels.append(data['bb_low'].iloc[-1])
        
        return sorted(list(set(key_levels))) 