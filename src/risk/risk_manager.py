"""
Gerenciador de risco
"""

from typing import Dict, Optional, List
import logging
from datetime import datetime
import numpy as np
import asyncio
from ..exchanges.order_manager import OrderManager, OrderConfig
from ..risk.stop_loss_manager import StopLossManager, StopLossConfig
from ..config.settings import (
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    MARKET_TYPE
)

class RiskManager:
    """Gerenciador de risco"""
    
    def __init__(
        self,
        order_manager: OrderManager,
        stop_loss_manager: StopLossManager
    ):
        self.order_manager = order_manager
        self.stop_loss_manager = stop_loss_manager
        self.logger = self._setup_logger()
        
        # Métricas de risco
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.trade_history: List[Dict] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Configura logger"""
        logger = logging.getLogger('RiskManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def calculate_position_size(
        self,
        symbol: str,
        price: float,
        balance: float,
        volatility: float
    ) -> float:
        """
        Calcula tamanho da posição baseado em risco
        
        Args:
            symbol: Par de trading
            price: Preço atual
            balance: Saldo disponível
            volatility: Volatilidade do ativo
            
        Returns:
            float com tamanho da posição
        """
        try:
            # Calcula risco máximo por trade (2% do capital)
            max_risk = balance * 0.02
            
            # Ajusta risco baseado na volatilidade
            adjusted_risk = max_risk * (1 - volatility)
            
            # Calcula tamanho da posição
            position_size = adjusted_risk / price
            
            # Limita tamanho da posição
            max_size = balance * MAX_POSITION_SIZE / price
            min_size = balance * MIN_POSITION_SIZE / price
            
            position_size = min(position_size, max_size)
            position_size = max(position_size, min_size)
            
            self.logger.info(
                f"Position size calculated for {symbol}: "
                f"{position_size:.8f} @ {price:.2f}"
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            raise
            
    async def can_open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        balance: float
    ) -> bool:
        """
        Verifica se pode abrir posição
        
        Args:
            symbol: Par de trading
            side: Lado da posição
            price: Preço atual
            amount: Quantidade
            balance: Saldo disponível
            
        Returns:
            bool indicando se pode abrir posição
        """
        try:
            # Verifica drawdown máximo
            if self.max_drawdown > 0.1:  # 10%
                self.logger.warning("Maximum drawdown exceeded")
                return False
                
            # Verifica perda diária
            if self.daily_pnl < -balance * 0.05:  # 5%
                self.logger.warning("Daily loss limit exceeded")
                return False
                
            # Verifica exposição total
            total_exposure = sum(
                trade['amount'] * trade['price']
                for trade in self.trade_history
                if trade['status'] == 'open'
            )
            
            if total_exposure > balance * 0.5:  # 50%
                self.logger.warning("Maximum exposure exceeded")
                return False
                
            # Verifica exposição por símbolo
            symbol_exposure = sum(
                trade['amount'] * trade['price']
                for trade in self.trade_history
                if trade['symbol'] == symbol and trade['status'] == 'open'
            )
            
            if symbol_exposure > balance * 0.2:  # 20%
                self.logger.warning("Maximum symbol exposure exceeded")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position: {str(e)}")
            return False
            
    async def add_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        stop_loss: float,
        take_profit: float
    ) -> None:
        """
        Adiciona trade ao histórico
        
        Args:
            symbol: Par de trading
            side: Lado da posição
            price: Preço de entrada
            amount: Quantidade
            stop_loss: Preço do stop loss
            take_profit: Preço do take profit
        """
        try:
            trade = {
                'symbol': symbol,
                'side': side,
                'price': price,
                'amount': amount,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open',
                'entry_time': datetime.now(),
                'pnl': 0.0
            }
            
            self.trade_history.append(trade)
            
            self.logger.info(
                f"Trade added: {symbol} {side} {amount:.8f} @ {price:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error adding trade: {str(e)}")
            raise
            
    async def update_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: datetime
    ) -> None:
        """
        Atualiza trade no histórico
        
        Args:
            trade_id: ID do trade
            exit_price: Preço de saída
            exit_time: Tempo de saída
        """
        try:
            if trade_id >= len(self.trade_history):
                raise ValueError(f"Trade {trade_id} not found")
                
            trade = self.trade_history[trade_id]
            
            # Calcula P&L
            if trade['side'] == 'buy':
                pnl = (exit_price - trade['price']) * trade['amount']
            else:
                pnl = (trade['price'] - exit_price) * trade['amount']
                
            # Atualiza trade
            trade['status'] = 'closed'
            trade['exit_price'] = exit_price
            trade['exit_time'] = exit_time
            trade['pnl'] = pnl
            
            # Atualiza métricas
            self.daily_pnl += pnl
            self.peak_balance = max(self.peak_balance, pnl)
            self.max_drawdown = min(
                self.max_drawdown,
                (pnl - self.peak_balance) / self.peak_balance
            )
            
            self.logger.info(
                f"Trade updated: {trade['symbol']} {trade['side']} "
                f"P&L: {pnl:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating trade: {str(e)}")
            raise
            
    def get_risk_metrics(self) -> Dict:
        """
        Obtém métricas de risco
        
        Returns:
            Dict com métricas de risco
        """
        try:
            return {
                'daily_pnl': self.daily_pnl,
                'max_drawdown': self.max_drawdown,
                'peak_balance': self.peak_balance,
                'total_trades': len(self.trade_history),
                'open_trades': len([
                    trade for trade in self.trade_history
                    if trade['status'] == 'open'
                ]),
                'win_rate': len([
                    trade for trade in self.trade_history
                    if trade['status'] == 'closed' and trade['pnl'] > 0
                ]) / len([
                    trade for trade in self.trade_history
                    if trade['status'] == 'closed'
                ]) if self.trade_history else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {str(e)}")
            raise 