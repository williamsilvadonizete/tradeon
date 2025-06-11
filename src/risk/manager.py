import sys
import os
from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging

# Adiciona o diretório raiz ao path para importar configurações
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    STOP_LOSS_PERCENTAGE,
    TAKE_PROFIT_PERCENTAGE,
    MAX_POSITION_SIZE
)

@dataclass
class RiskMetrics:
    win_rate: float
    profit_ratio: float
    max_drawdown: float
    current_drawdown: float
    peak_equity: float
    current_equity: float

class RiskManager:
    def __init__(self, config: Dict):
        self.max_drawdown = config.get('max_drawdown', 0.1)  # 10% máximo de drawdown
        self.trailing_stop = config.get('trailing_stop', 0.02)  # 2% trailing stop
        self.kelly_fraction = config.get('kelly_fraction', 0.5)  # Usar metade do Kelly Criterion
        self.max_position_size = config.get('max_position_size', 0.1)  # Máximo 10% do capital
        self.min_position_size = config.get('min_position_size', 0.01)  # Mínimo 1% do capital
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% de risco por trade
        self.logger = logging.getLogger(__name__)
        
        # Métricas de performance
        self.metrics = RiskMetrics(
            win_rate=0.0,
            profit_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            peak_equity=0.0,
            current_equity=0.0
        )
        
    def calculate_position_size(self, win_rate: float, profit_ratio: float, 
                              current_equity: float, volatility: float) -> float:
        """
        Calcula o tamanho da posição usando Kelly Criterion e ajustando pela volatilidade.
        
        Args:
            win_rate: Taxa de acerto histórica
            profit_ratio: Razão entre ganhos e perdas
            current_equity: Capital atual
            volatility: Volatilidade atual do mercado
            
        Returns:
            float: Tamanho da posição como fração do capital
        """
        try:
            # Kelly Criterion básico
            kelly = (win_rate * profit_ratio - (1 - win_rate)) / profit_ratio
            
            # Ajuste pela volatilidade (maior volatilidade = menor posição)
            volatility_factor = 1 / (1 + volatility)
            
            # Ajuste final
            position_size = kelly * self.kelly_fraction * volatility_factor
            
            # Limites
            position_size = max(min(position_size, self.max_position_size), self.min_position_size)
            
            self.logger.info(f"Position size calculated: {position_size:.4f} (Kelly: {kelly:.4f}, "
                           f"Volatility factor: {volatility_factor:.4f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return self.min_position_size
            
    def update_trailing_stop(self, current_price: float, highest_price: float, 
                           position_type: str) -> float:
        """
        Atualiza o trailing stop baseado no preço mais alto/baixo desde a entrada.
        
        Args:
            current_price: Preço atual
            highest_price: Preço mais alto desde a entrada
            position_type: Tipo de posição ('long' ou 'short')
            
        Returns:
            float: Novo nível de stop loss
        """
        try:
            if position_type == 'long':
                return highest_price * (1 - self.trailing_stop)
            else:
                return highest_price * (1 + self.trailing_stop)
                
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")
            return current_price
            
    def check_drawdown(self, current_equity: float) -> bool:
        """
        Verifica se o drawdown atual está dentro dos limites aceitáveis.
        
        Args:
            current_equity: Capital atual
            
        Returns:
            bool: True se o drawdown está aceitável, False caso contrário
        """
        try:
            # Atualiza métricas
            self.metrics.current_equity = current_equity
            self.metrics.peak_equity = max(self.metrics.peak_equity, current_equity)
            
            # Calcula drawdown atual
            if self.metrics.peak_equity > 0:
                self.metrics.current_drawdown = (self.metrics.peak_equity - current_equity) / self.metrics.peak_equity
                self.metrics.max_drawdown = max(self.metrics.max_drawdown, self.metrics.current_drawdown)
            
            # Verifica limite
            is_acceptable = self.metrics.current_drawdown <= self.max_drawdown
            
            if not is_acceptable:
                self.logger.warning(f"Drawdown limit exceeded: {self.metrics.current_drawdown:.2%} "
                                  f"(max: {self.max_drawdown:.2%})")
            
            return is_acceptable
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {str(e)}")
            return False
            
    def update_metrics(self, trade_result: Dict):
        """
        Atualiza as métricas de performance com o resultado de um trade.
        
        Args:
            trade_result: Dicionário com o resultado do trade
        """
        try:
            # Atualiza win rate
            total_trades = trade_result.get('total_trades', 0)
            winning_trades = trade_result.get('winning_trades', 0)
            if total_trades > 0:
                self.metrics.win_rate = winning_trades / total_trades
            
            # Atualiza profit ratio
            total_profit = trade_result.get('total_profit', 0)
            total_loss = abs(trade_result.get('total_loss', 0))
            if total_loss > 0:
                self.metrics.profit_ratio = total_profit / total_loss
            
            self.logger.info(f"Metrics updated - Win rate: {self.metrics.win_rate:.2%}, "
                           f"Profit ratio: {self.metrics.profit_ratio:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Retorna as métricas de risco atuais.
        
        Returns:
            RiskMetrics: Objeto com as métricas de risco
        """
        return self.metrics

    def calculate_stop_loss(self, entry_price, is_long=True):
        """
        Calcula o preço de stop loss.
        
        Args:
            entry_price (float): Preço de entrada
            is_long (bool): Se a posição é comprada
            
        Returns:
            float: Preço de stop loss
        """
        if is_long:
            return entry_price * (1 - STOP_LOSS_PERCENTAGE / 100)
        else:
            return entry_price * (1 + STOP_LOSS_PERCENTAGE / 100)
            
    def calculate_take_profit(self, entry_price, is_long=True):
        """
        Calcula o preço de take profit.
        
        Args:
            entry_price (float): Preço de entrada
            is_long (bool): Se a posição é comprada
            
        Returns:
            float: Preço de take profit
        """
        if is_long:
            return entry_price * (1 + TAKE_PROFIT_PERCENTAGE / 100)
        else:
            return entry_price * (1 - TAKE_PROFIT_PERCENTAGE / 100)
            
    def update_position(self, symbol, entry_price, position_size, is_long=True):
        """
        Atualiza o registro de posições.
        
        Args:
            symbol (str): Par de trading
            entry_price (float): Preço de entrada
            position_size (float): Tamanho da posição
            is_long (bool): Se a posição é comprada
        """
        self.positions[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'is_long': is_long,
            'stop_loss': self.calculate_stop_loss(entry_price, is_long),
            'take_profit': self.calculate_take_profit(entry_price, is_long)
        }
        
    def check_stop_loss_take_profit(self, symbol, current_price):
        """
        Verifica se alguma condição de stop loss ou take profit foi atingida.
        
        Args:
            symbol (str): Par de trading
            current_price (float): Preço atual
            
        Returns:
            str: 'stop_loss', 'take_profit' ou None
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        if position['is_long']:
            if current_price <= position['stop_loss']:
                return 'stop_loss'
            elif current_price >= position['take_profit']:
                return 'take_profit'
        else:
            if current_price >= position['stop_loss']:
                return 'stop_loss'
            elif current_price <= position['take_profit']:
                return 'take_profit'
                
        return None
        
    def close_position(self, symbol, current_price):
        """
        Fecha uma posição e atualiza o capital.
        
        Args:
            symbol (str): Par de trading
            current_price (float): Preço atual
            
        Returns:
            float: Lucro/Prejuízo da operação
        """
        if symbol not in self.positions:
            return 0
            
        position = self.positions[symbol]
        entry_value = position['entry_price'] * position['position_size']
        exit_value = current_price * position['position_size']
        
        if position['is_long']:
            pnl = exit_value - entry_value
        else:
            pnl = entry_value - exit_value
            
        self.current_capital += pnl
        del self.positions[symbol]
        
        return pnl 