from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

@dataclass
class StopLossConfig:
    """Configuração para diferentes estratégias de stop loss."""
    strategy: str  # 'fixed', 'atr', 'support', 'trailing'
    atr_period: int = 14
    atr_multiplier: float = 2.0
    fixed_percentage: float = 0.02  # 2%
    trailing_activation: float = 0.01  # 1%
    trailing_distance: float = 0.005  # 0.5%
    support_lookback: int = 20

class StopLossManager:
    """Gerencia diferentes estratégias de stop loss."""
    
    def __init__(self, config: StopLossConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_stop_loss(self, 
                          symbol: str,
                          position_side: str,
                          entry_price: float,
                          historical_data: pd.DataFrame) -> float:
        """
        Calcula o preço do stop loss baseado na estratégia configurada.
        
        Args:
            symbol: Símbolo do par de trading
            position_side: 'long' ou 'short'
            entry_price: Preço de entrada
            historical_data: DataFrame com dados históricos
            
        Returns:
            float: Preço do stop loss
        """
        try:
            if self.config.strategy == 'fixed':
                return self._calculate_fixed_stop(entry_price, position_side)
            elif self.config.strategy == 'atr':
                return self._calculate_atr_stop(historical_data, entry_price, position_side)
            elif self.config.strategy == 'support':
                return self._calculate_support_stop(historical_data, entry_price, position_side)
            elif self.config.strategy == 'trailing':
                return self._calculate_trailing_stop(historical_data, entry_price, position_side)
            else:
                raise ValueError(f"Estratégia de stop loss desconhecida: {self.config.strategy}")
                
        except Exception as e:
            self.logger.error(f"Erro ao calcular stop loss: {str(e)}")
            # Fallback para stop fixo em caso de erro
            return self._calculate_fixed_stop(entry_price, position_side)
    
    def _calculate_fixed_stop(self, entry_price: float, position_side: str) -> float:
        """Calcula stop loss com porcentagem fixa."""
        if position_side == 'long':
            return entry_price * (1 - self.config.fixed_percentage)
        else:
            return entry_price * (1 + self.config.fixed_percentage)
    
    def _calculate_atr_stop(self, 
                           data: pd.DataFrame,
                           entry_price: float,
                           position_side: str) -> float:
        """Calcula stop loss baseado no ATR."""
        # Calcula ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(self.config.atr_period).mean().iloc[-1]
        
        # Calcula stop loss
        if position_side == 'long':
            return entry_price - (atr * self.config.atr_multiplier)
        else:
            return entry_price + (atr * self.config.atr_multiplier)
    
    def _calculate_support_stop(self,
                              data: pd.DataFrame,
                              entry_price: float,
                              position_side: str) -> float:
        """Calcula stop loss baseado em níveis de suporte/resistência."""
        if position_side == 'long':
            # Encontra nível de suporte mais próximo
            support_levels = self._find_support_levels(data)
            if support_levels:
                return max(support_levels[-1], entry_price * 0.95)  # Limite mínimo de 5%
            return entry_price * 0.95
        else:
            # Encontra nível de resistência mais próximo
            resistance_levels = self._find_resistance_levels(data)
            if resistance_levels:
                return min(resistance_levels[-1], entry_price * 1.05)  # Limite máximo de 5%
            return entry_price * 1.05
    
    def _calculate_trailing_stop(self,
                               data: pd.DataFrame,
                               entry_price: float,
                               position_side: str) -> float:
        """Configura trailing stop."""
        current_price = data['close'].iloc[-1]
        
        if position_side == 'long':
            if current_price > entry_price * (1 + self.config.trailing_activation):
                return current_price * (1 - self.config.trailing_distance)
            return entry_price * (1 - self.config.fixed_percentage)
        else:
            if current_price < entry_price * (1 - self.config.trailing_activation):
                return current_price * (1 + self.config.trailing_distance)
            return entry_price * (1 + self.config.fixed_percentage)
    
    def _find_support_levels(self, data: pd.DataFrame) -> list:
        """Encontra níveis de suporte no histórico."""
        lows = data['low'].rolling(window=5, center=True).min()
        support_levels = []
        
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        return sorted(support_levels)
    
    def _find_resistance_levels(self, data: pd.DataFrame) -> list:
        """Encontra níveis de resistência no histórico."""
        highs = data['high'].rolling(window=5, center=True).max()
        resistance_levels = []
        
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
        
        return sorted(resistance_levels)
    
    def should_update_stop(self,
                          current_price: float,
                          current_stop: float,
                          position_side: str) -> bool:
        """
        Verifica se o stop loss deve ser atualizado.
        
        Args:
            current_price: Preço atual
            current_stop: Stop loss atual
            position_side: 'long' ou 'short'
            
        Returns:
            bool: True se o stop deve ser atualizado
        """
        if self.config.strategy != 'trailing':
            return False
            
        if position_side == 'long':
            return current_price > current_stop * (1 + self.config.trailing_activation)
        else:
            return current_price < current_stop * (1 - self.config.trailing_activation) 