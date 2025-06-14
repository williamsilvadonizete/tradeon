from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
from ..exchanges.order_manager import OrderManager, OrderConfig
from ..config.settings import (
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    MARKET_TYPE
)

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
    
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.logger = self._setup_logger()
        self.active_stops: Dict[str, Dict] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Configura logger"""
        logger = logging.getLogger('StopLossManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def setup_stop_loss(self, config: StopLossConfig) -> Dict:
        """
        Configura stop loss para uma posição
        
        Args:
            config: Configuração do stop loss
            
        Returns:
            Dict com informações do stop loss
        """
        try:
            # Calcula preços se não fornecidos
            if not config.stop_loss_price:
                if config.side == 'buy':
                    config.stop_loss_price = config.entry_price * (
                        1 - STOP_LOSS_PERCENT
                    )
                else:
                    config.stop_loss_price = config.entry_price * (
                        1 + STOP_LOSS_PERCENT
                    )
                    
            if not config.take_profit_price:
                if config.side == 'buy':
                    config.take_profit_price = config.entry_price * (
                        1 + TAKE_PROFIT_PERCENT
                    )
                else:
                    config.take_profit_price = config.entry_price * (
                        1 - TAKE_PROFIT_PERCENT
                    )
                    
            # Cria ordem de stop loss
            sl_config = OrderConfig(
                symbol=config.symbol,
                side='sell' if config.side == 'buy' else 'buy',
                type='stop',
                amount=config.amount,
                stop_price=config.stop_loss_price
            )
            
            sl_order = await self.order_manager.create_order(sl_config)
            
            # Cria ordem de take profit
            tp_config = OrderConfig(
                symbol=config.symbol,
                side='sell' if config.side == 'buy' else 'buy',
                type='limit',
                amount=config.amount,
                price=config.take_profit_price
            )
            
            tp_order = await self.order_manager.create_order(tp_config)
            
            # Configura trailing stop se especificado
            if config.trailing_stop:
                await self._setup_trailing_stop(config)
                
            # Registra stops ativos
            self.active_stops[sl_order['id']] = {
                'symbol': config.symbol,
                'side': config.side,
                'amount': config.amount,
                'entry_price': config.entry_price,
                'stop_loss': sl_order,
                'take_profit': tp_order,
                'trailing_stop': config.trailing_stop,
                'trailing_activation': config.trailing_activation,
                'created_at': datetime.now()
            }
            
            return {
                'stop_loss': sl_order,
                'take_profit': tp_order
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up stop loss: {str(e)}")
            raise
            
    async def _setup_trailing_stop(self, config: StopLossConfig) -> None:
        """Configura trailing stop"""
        try:
            # Calcula preço de ativação se não fornecido
            if not config.trailing_activation:
                if config.side == 'buy':
                    config.trailing_activation = config.entry_price * (
                        1 + config.trailing_stop
                    )
                else:
                    config.trailing_activation = config.entry_price * (
                        1 - config.trailing_stop
                    )
                    
            # Cria ordem de trailing stop
            trailing_config = OrderConfig(
                symbol=config.symbol,
                side='sell' if config.side == 'buy' else 'buy',
                type='trailing_stop',
                amount=config.amount,
                stop_price=config.trailing_activation
            )
            
            trailing_order = await self.order_manager.create_order(trailing_config)
            
            self.logger.info(
                f"Trailing stop set for {config.symbol}: "
                f"activation at {config.trailing_activation}, "
                f"trailing by {config.trailing_stop}"
            )
            
        except Exception as e:
            self.logger.error(f"Error setting up trailing stop: {str(e)}")
            raise
            
    async def update_stop_loss(
        self,
        stop_id: str,
        new_stop_price: float
    ) -> Dict:
        """
        Atualiza preço de stop loss
        
        Args:
            stop_id: ID do stop loss
            new_stop_price: Novo preço do stop
            
        Returns:
            Dict com stop loss atualizado
        """
        try:
            if stop_id not in self.active_stops:
                raise ValueError(f"Stop loss {stop_id} not found")
                
            stop_info = self.active_stops[stop_id]
            
            # Cancela stop loss antigo
            await self.order_manager.cancel_order(
                stop_info['stop_loss']['id'],
                stop_info['symbol']
            )
            
            # Cria novo stop loss
            sl_config = OrderConfig(
                symbol=stop_info['symbol'],
                side='sell' if stop_info['side'] == 'buy' else 'buy',
                type='stop',
                amount=stop_info['amount'],
                stop_price=new_stop_price
            )
            
            new_sl = await self.order_manager.create_order(sl_config)
            
            # Atualiza registro
            self.active_stops[stop_id]['stop_loss'] = new_sl
            
            self.logger.info(
                f"Stop loss updated for {stop_info['symbol']}: "
                f"new price {new_stop_price}"
            )
            
            return new_sl
            
        except Exception as e:
            self.logger.error(f"Error updating stop loss: {str(e)}")
            raise
            
    async def cancel_stop_loss(self, stop_id: str) -> None:
        """
        Cancela stop loss
        
        Args:
            stop_id: ID do stop loss
        """
        try:
            if stop_id not in self.active_stops:
                raise ValueError(f"Stop loss {stop_id} not found")
                
            stop_info = self.active_stops[stop_id]
            
            # Cancela stop loss e take profit
            await self.order_manager.cancel_order(
                stop_info['stop_loss']['id'],
                stop_info['symbol']
            )
            
            await self.order_manager.cancel_order(
                stop_info['take_profit']['id'],
                stop_info['symbol']
            )
            
            # Remove registro
            del self.active_stops[stop_id]
            
            self.logger.info(f"Stop loss cancelled for {stop_info['symbol']}")
            
        except Exception as e:
            self.logger.error(f"Error cancelling stop loss: {str(e)}")
            raise
            
    async def get_active_stops(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtém stops ativos
        
        Args:
            symbol: Par de trading (opcional)
            
        Returns:
            List[Dict] com stops ativos
        """
        try:
            if symbol:
                return [
                    stop for stop in self.active_stops.values()
                    if stop['symbol'] == symbol
                ]
            return list(self.active_stops.values())
            
        except Exception as e:
            self.logger.error(f"Error getting active stops: {str(e)}")
            raise
    
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