"""
Advanced Backtesting System

This module implements a comprehensive backtesting framework for trading strategies
with realistic market simulation, slippage, fees, and detailed performance analysis.

Author: Trading Bot System
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from ..strategies.base import TradingStrategy
from ..strategies.bullish_strategies import BullishStrategyManager
from ..strategies.bearish_strategies import BearishStrategyManager
from ..risk.advanced_risk_manager import AdvancedRiskManager
from ..analysis.multitimeframe import MultiTimeframeAnalyzer

class OrderType(Enum):
    """Tipos de ordem"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Status da ordem"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    """Lado da posição"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    """Ordem de trading"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    filled_timestamp: Optional[datetime] = None
    strategy_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class Position:
    """Posição de trading"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    entry_timestamp: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    strategy_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class Trade:
    """Trade executado"""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    pnl: float
    fees: float
    duration_minutes: float
    strategy_id: Optional[str] = None
    entry_reason: str = ""
    exit_reason: str = ""
    metadata: Dict = field(default_factory=dict)

@dataclass
class BacktestConfig:
    """Configuração do backtest"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    max_positions: int = 10
    position_sizing_method: str = "fixed_percent"  # 'fixed_percent', 'risk_based', 'kelly'
    default_position_size: float = 0.1  # 10% do capital
    margin_requirement: float = 1.0  # 100% = sem alavancagem
    interest_rate: float = 0.0  # Taxa de juros para posições short
    risk_free_rate: float = 0.02  # Taxa livre de risco anual
    benchmark_symbol: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timeframe: str = "5m"
    
@dataclass
class BacktestMetrics:
    """Métricas de performance do backtest"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration: float = 0.0
    total_fees: float = 0.0
    calmar_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0

class MarketSimulator:
    """Simulador de mercado para backtesting realístico"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estado do mercado
        self.current_time = None
        self.current_prices = {}
        self.price_history = {}
        self.volume_history = {}
        
        # Liquidez e spread simulation
        self.spread_config = {
            'base_spread': 0.0005,  # 0.05% spread base
            'volume_impact': 0.0001,  # Impacto do volume
            'volatility_impact': 0.0002  # Impacto da volatilidade
        }
    
    def update_market_state(self, timestamp: datetime, market_data: Dict[str, pd.DataFrame]):
        """Atualiza estado do mercado"""
        self.current_time = timestamp
        
        for symbol, data in market_data.items():
            # Encontra a linha correspondente ao timestamp
            try:
                current_row = data.loc[data.index <= timestamp].iloc[-1]
                self.current_prices[symbol] = {
                    'open': current_row['open'],
                    'high': current_row['high'],
                    'low': current_row['low'],
                    'close': current_row['close'],
                    'volume': current_row['volume']
                }
                
                # Histórico para cálculos
                recent_data = data.loc[data.index <= timestamp].tail(20)
                self.price_history[symbol] = recent_data['close'].tolist()
                self.volume_history[symbol] = recent_data['volume'].tolist()
                
            except (IndexError, KeyError):
                continue
    
    def calculate_execution_price(self, symbol: str, side: str, quantity: float, 
                                order_type: OrderType, limit_price: Optional[float] = None) -> Optional[float]:
        """Calcula preço de execução considerando slippage e spread"""
        if symbol not in self.current_prices:
            return None
        
        market_price = self.current_prices[symbol]['close']
        
        # Calcula spread baseado em condições de mercado
        spread = self._calculate_dynamic_spread(symbol, quantity)
        
        # Slippage baseado no tamanho da ordem
        slippage = self._calculate_slippage(symbol, quantity)
        
        if order_type == OrderType.MARKET:
            if side == 'buy':
                execution_price = market_price * (1 + spread/2 + slippage)
            else:
                execution_price = market_price * (1 - spread/2 - slippage)
        
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                return None
            
            # Verifica se a ordem limit seria executada
            if side == 'buy' and limit_price >= market_price * (1 + spread/2):
                execution_price = min(limit_price, market_price * (1 + spread/2))
            elif side == 'sell' and limit_price <= market_price * (1 - spread/2):
                execution_price = max(limit_price, market_price * (1 - spread/2))
            else:
                return None  # Ordem não seria executada
        
        else:
            # Para outros tipos de ordem, usa preço de mercado por simplicidade
            execution_price = market_price
        
        return execution_price
    
    def _calculate_dynamic_spread(self, symbol: str, quantity: float) -> float:
        """Calcula spread dinâmico baseado em condições de mercado"""
        base_spread = self.spread_config['base_spread']
        
        # Impacto da volatilidade
        if symbol in self.price_history and len(self.price_history[symbol]) > 10:
            returns = np.diff(self.price_history[symbol]) / self.price_history[symbol][:-1]
            volatility = np.std(returns)
            volatility_impact = volatility * self.spread_config['volatility_impact']
        else:
            volatility_impact = 0
        
        # Impacto do volume da ordem
        if symbol in self.volume_history and len(self.volume_history[symbol]) > 0:
            avg_volume = np.mean(self.volume_history[symbol])
            volume_impact = (quantity / avg_volume) * self.spread_config['volume_impact']
        else:
            volume_impact = 0
        
        total_spread = base_spread + volatility_impact + volume_impact
        return min(total_spread, 0.01)  # Limita spread a 1%
    
    def _calculate_slippage(self, symbol: str, quantity: float) -> float:
        """Calcula slippage baseado no tamanho da ordem"""
        base_slippage = self.config.slippage_rate
        
        # Slippage adicional para ordens grandes
        if symbol in self.volume_history and len(self.volume_history[symbol]) > 0:
            avg_volume = np.mean(self.volume_history[symbol])
            if avg_volume > 0:
                size_factor = quantity / avg_volume
                additional_slippage = base_slippage * size_factor * 0.5
                return base_slippage + additional_slippage
        
        return base_slippage

class AdvancedBacktester:
    """Sistema avançado de backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Componentes
        self.market_simulator = MarketSimulator(config)
        self.risk_manager = AdvancedRiskManager()
        self.multitimeframe_analyzer = MultiTimeframeAnalyzer()
        
        # Estado do backtest
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}  # symbol -> Position
        self.orders = {}  # order_id -> Order
        self.trades = []  # Lista de trades completos
        self.portfolio_history = []  # Histórico do valor do portfólio
        
        # Métricas
        self.daily_returns = []
        self.benchmark_returns = []
        
        # Contadores
        self.order_counter = 0
        self.trade_counter = 0
        
    def run_backtest(self, strategies: Dict[str, TradingStrategy], 
                    market_data: Dict[str, pd.DataFrame],
                    benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Executa backtest completo
        
        Args:
            strategies: Dicionário de estratégias {nome: estratégia}
            market_data: Dados de mercado {símbolo: DataFrame}
            benchmark_data: Dados do benchmark (opcional)
            
        Returns:
            Resultados completos do backtest
        """
        self.logger.info("Iniciando backtest avançado...")
        
        # Preparação
        self._initialize_backtest(market_data, benchmark_data)
        
        # Timeline do backtest
        all_timestamps = self._create_timeline(market_data)
        
        # Executa backtest timestamp por timestamp
        for i, timestamp in enumerate(all_timestamps):
            if i % 1000 == 0:
                self.logger.info(f"Processando {i}/{len(all_timestamps)} ({timestamp})")
            
            # Atualiza estado do mercado
            self._update_market_state(timestamp, market_data)
            
            # Processa ordens pendentes
            self._process_pending_orders()
            
            # Atualiza posições
            self._update_positions()
            
            # Gera sinais das estratégias
            self._generate_strategy_signals(strategies, market_data, timestamp)
            
            # Atualiza métricas
            self._update_portfolio_metrics(timestamp, benchmark_data)
        
        # Fecha posições remanescentes
        self._close_remaining_positions()
        
        # Calcula métricas finais
        metrics = self._calculate_final_metrics()
        
        # Gera relatório
        results = self._generate_backtest_report(strategies, metrics)
        
        self.logger.info("Backtest concluído!")
        return results
    
    def _initialize_backtest(self, market_data: Dict[str, pd.DataFrame], 
                           benchmark_data: Optional[pd.DataFrame]):
        """Inicializa estado do backtest"""
        # Reset estado
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.benchmark_returns = []
        
        # Valida dados
        for symbol, data in market_data.items():
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Dados incompletos para {symbol}")
    
    def _create_timeline(self, market_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Cria timeline unificada para o backtest"""
        all_timestamps = set()
        
        for data in market_data.values():
            all_timestamps.update(data.index)
        
        timeline = sorted(list(all_timestamps))
        
        # Filtra por período se especificado
        if self.config.start_date:
            timeline = [ts for ts in timeline if ts >= self.config.start_date]
        if self.config.end_date:
            timeline = [ts for ts in timeline if ts <= self.config.end_date]
        
        return timeline
    
    def _update_market_state(self, timestamp: datetime, market_data: Dict[str, pd.DataFrame]):
        """Atualiza estado do mercado"""
        self.market_simulator.update_market_state(timestamp, market_data)
    
    def _process_pending_orders(self):
        """Processa ordens pendentes"""
        for order_id, order in list(self.orders.items()):
            if order.status == OrderStatus.PENDING:
                execution_price = self.market_simulator.calculate_execution_price(
                    order.symbol, order.side, order.quantity, order.order_type, order.price
                )
                
                if execution_price is not None:
                    self._execute_order(order, execution_price)
    
    def _execute_order(self, order: Order, execution_price: float):
        """Executa uma ordem"""
        try:
            # Calcula custos
            trade_value = order.quantity * execution_price
            commission = trade_value * self.config.commission_rate
            
            # Verifica disponibilidade de capital/posições
            if order.side == 'buy':
                total_cost = trade_value + commission
                if total_cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    order.metadata['rejection_reason'] = 'Insufficient cash'
                    return
                
                self.cash -= total_cost
            else:  # sell
                # Verifica se tem posição para vender
                if order.symbol not in self.positions:
                    order.status = OrderStatus.REJECTED
                    order.metadata['rejection_reason'] = 'No position to sell'
                    return
                
                self.cash += trade_value - commission
            
            # Atualiza ordem
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.filled_timestamp = self.market_simulator.current_time
            
            # Atualiza ou cria posição
            self._update_position_from_order(order, execution_price, commission)
            
        except Exception as e:
            self.logger.error(f"Erro executando ordem {order.id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = str(e)
    
    def _update_position_from_order(self, order: Order, execution_price: float, commission: float):
        """Atualiza posição baseada na ordem executada"""
        symbol = order.symbol
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if order.side == 'buy' and position.side == PositionSide.LONG:
                # Aumenta posição long
                total_value = position.quantity * position.entry_price + order.quantity * execution_price
                position.quantity += order.quantity
                position.entry_price = total_value / position.quantity
                position.fees_paid += commission
                
            elif order.side == 'sell' and position.side == PositionSide.LONG:
                # Reduz ou fecha posição long
                if order.quantity >= position.quantity:
                    # Fecha posição completamente
                    pnl = (execution_price - position.entry_price) * position.quantity
                    self._close_position(position, execution_price, commission, pnl, "Full exit")
                else:
                    # Reduz posição
                    pnl = (execution_price - position.entry_price) * order.quantity
                    position.realized_pnl += pnl
                    position.quantity -= order.quantity
                    position.fees_paid += commission
            
            # TODO: Implementar lógica para posições short
            
        else:
            # Nova posição
            if order.side == 'buy':
                side = PositionSide.LONG
            else:
                side = PositionSide.SHORT
                
            position = Position(
                symbol=symbol,
                side=side,
                quantity=order.quantity,
                entry_price=execution_price,
                current_price=execution_price,
                entry_timestamp=self.market_simulator.current_time,
                fees_paid=commission,
                strategy_id=order.strategy_id
            )
            
            self.positions[symbol] = position
    
    def _close_position(self, position: Position, exit_price: float, 
                       commission: float, pnl: float, reason: str):
        """Fecha uma posição e cria trade record"""
        # Cria trade record
        trade = Trade(
            id=f"T{self.trade_counter:06d}",
            symbol=position.symbol,
            side="long" if position.side == PositionSide.LONG else "short",
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=self.market_simulator.current_time,
            pnl=pnl,
            fees=position.fees_paid + commission,
            duration_minutes=(self.market_simulator.current_time - position.entry_timestamp).total_seconds() / 60,
            strategy_id=position.strategy_id,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        self.trade_counter += 1
        
        # Remove posição
        del self.positions[position.symbol]
    
    def _update_positions(self):
        """Atualiza posições com preços atuais"""
        for symbol, position in self.positions.items():
            if symbol in self.market_simulator.current_prices:
                current_price = self.market_simulator.current_prices[symbol]['close']
                position.current_price = current_price
                
                # Calcula PnL não realizado
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
    
    def _generate_strategy_signals(self, strategies: Dict[str, TradingStrategy], 
                                 market_data: Dict[str, pd.DataFrame], 
                                 timestamp: datetime):
        """Gera sinais das estratégias"""
        for strategy_name, strategy in strategies.items():
            try:
                for symbol, data in market_data.items():
                    # Obtém dados até o timestamp atual
                    historical_data = data.loc[data.index <= timestamp]
                    
                    if len(historical_data) < 50:  # Dados insuficientes
                        continue
                    
                    # Gera sinal
                    signal = strategy.generate_signal(historical_data)
                    
                    if signal and signal.get('action') in ['buy', 'sell']:
                        # Valida com risk manager
                        if self._validate_signal_with_risk_manager(signal, symbol):
                            # Calcula tamanho da posição
                            position_size = self._calculate_position_size(signal, symbol)
                            
                            if position_size > 0:
                                # Cria ordem
                                order = self._create_order_from_signal(
                                    signal, symbol, position_size, strategy_name
                                )
                                self.orders[order.id] = order
                
            except Exception as e:
                self.logger.error(f"Erro gerando sinais para {strategy_name}: {str(e)}")
    
    def _validate_signal_with_risk_manager(self, signal: Dict, symbol: str) -> bool:
        """Valida sinal com risk manager"""
        # Verifica limites básicos
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Verifica se já tem posição no símbolo
        if symbol in self.positions:
            return False
        
        # TODO: Implementar validações mais avançadas do risk manager
        return True
    
    def _calculate_position_size(self, signal: Dict, symbol: str) -> float:
        """Calcula tamanho da posição"""
        if symbol not in self.market_simulator.current_prices:
            return 0
        
        current_price = self.market_simulator.current_prices[symbol]['close']
        
        if self.config.position_sizing_method == "fixed_percent":
            position_value = self.portfolio_value * self.config.default_position_size
            quantity = position_value / current_price
        
        elif self.config.position_sizing_method == "risk_based":
            # Baseado no stop loss do sinal
            stop_loss = signal.get('stop_loss', current_price * 0.95)
            risk_per_share = abs(current_price - stop_loss)
            
            if risk_per_share > 0:
                max_risk = self.portfolio_value * 0.02  # 2% de risco máximo
                quantity = max_risk / risk_per_share
            else:
                quantity = 0
        
        else:
            # Fallback para tamanho fixo
            quantity = self.portfolio_value * 0.1 / current_price
        
        return max(0, quantity)
    
    def _create_order_from_signal(self, signal: Dict, symbol: str, 
                                quantity: float, strategy_name: str) -> Order:
        """Cria ordem baseada no sinal"""
        order_id = f"O{self.order_counter:06d}"
        self.order_counter += 1
        
        return Order(
            id=order_id,
            symbol=symbol,
            side=signal['action'],
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=self.market_simulator.current_time,
            strategy_id=strategy_name,
            metadata={
                'signal_confidence': signal.get('confidence', 0),
                'signal_strength': signal.get('strength', 0),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit')
            }
        )
    
    def _update_portfolio_metrics(self, timestamp: datetime, 
                                benchmark_data: Optional[pd.DataFrame]):
        """Atualiza métricas do portfólio"""
        # Calcula valor total do portfólio
        total_position_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        self.portfolio_value = self.cash + total_position_value
        
        # Salva histórico
        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': total_position_value,
            'num_positions': len(self.positions)
        })
        
        # Calcula retornos diários
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            daily_return = (self.portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
            
            # Benchmark returns
            if benchmark_data is not None:
                try:
                    current_benchmark = benchmark_data.loc[benchmark_data.index <= timestamp].iloc[-1]['close']
                    prev_benchmark = benchmark_data.loc[benchmark_data.index <= self.portfolio_history[-2]['timestamp']].iloc[-1]['close']
                    benchmark_return = (current_benchmark - prev_benchmark) / prev_benchmark
                    self.benchmark_returns.append(benchmark_return)
                except:
                    self.benchmark_returns.append(0)
    
    def _close_remaining_positions(self):
        """Fecha todas as posições remanescentes no final do backtest"""
        for symbol, position in list(self.positions.items()):
            if symbol in self.market_simulator.current_prices:
                exit_price = self.market_simulator.current_prices[symbol]['close']
                commission = exit_price * position.quantity * self.config.commission_rate
                
                if position.side == PositionSide.LONG:
                    pnl = (exit_price - position.entry_price) * position.quantity
                else:
                    pnl = (position.entry_price - exit_price) * position.quantity
                
                self._close_position(position, exit_price, commission, pnl, "End of backtest")
    
    def _calculate_final_metrics(self) -> BacktestMetrics:
        """Calcula métricas finais do backtest"""
        metrics = BacktestMetrics()
        
        if not self.portfolio_history or not self.trades:
            return metrics
        
        # Retornos básicos
        initial_value = self.config.initial_capital
        final_value = self.portfolio_history[-1]['portfolio_value']
        metrics.total_return = (final_value - initial_value) / initial_value
        
        # Duração do backtest
        start_date = self.portfolio_history[0]['timestamp']
        end_date = self.portfolio_history[-1]['timestamp']
        duration_years = (end_date - start_date).days / 365.25
        
        if duration_years > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (1/duration_years) - 1
        
        # Volatilidade
        if len(self.daily_returns) > 1:
            metrics.volatility = np.std(self.daily_returns) * np.sqrt(252)
            
            # Sharpe Ratio
            excess_returns = np.array(self.daily_returns) - self.config.risk_free_rate/252
            if np.std(excess_returns) > 0:
                metrics.sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            # Sortino Ratio
            negative_returns = excess_returns[excess_returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns) * np.sqrt(252)
                if downside_deviation > 0:
                    metrics.sortino_ratio = metrics.annualized_return / downside_deviation
        
        # Drawdown
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        metrics.max_drawdown = abs(np.min(drawdowns))
        
        # Calmar Ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
        
        # Métricas de trades
        if self.trades:
            pnls = [trade.pnl for trade in self.trades]
            metrics.total_trades = len(self.trades)
            
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl < 0]
            
            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            if winning_trades:
                metrics.avg_win = np.mean(winning_trades)
                metrics.largest_win = np.max(winning_trades)
            
            if losing_trades:
                metrics.avg_loss = np.mean(losing_trades)
                metrics.largest_loss = np.min(losing_trades)
            
            # Profit Factor
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 1
            metrics.profit_factor = total_wins / total_losses
            
            # Duração média dos trades
            durations = [trade.duration_minutes for trade in self.trades]
            metrics.avg_trade_duration = np.mean(durations)
            
            # Total de taxas
            metrics.total_fees = sum(trade.fees for trade in self.trades)
        
        # Beta e Alpha (se temos benchmark)
        if len(self.benchmark_returns) == len(self.daily_returns) and len(self.daily_returns) > 1:
            portfolio_returns = np.array(self.daily_returns)
            benchmark_returns = np.array(self.benchmark_returns)
            
            if np.var(benchmark_returns) > 0:
                metrics.beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                metrics.alpha = np.mean(portfolio_returns) - metrics.beta * np.mean(benchmark_returns)
                metrics.alpha *= 252  # Anualizada
        
        # VaR e Expected Shortfall
        if len(self.daily_returns) > 1:
            sorted_returns = np.sort(self.daily_returns)
            var_index = int(0.05 * len(sorted_returns))
            metrics.var_95 = sorted_returns[var_index] if var_index < len(sorted_returns) else 0
            
            tail_returns = sorted_returns[:var_index] if var_index > 0 else sorted_returns[:1]
            metrics.expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0
        
        return metrics
    
    def _generate_backtest_report(self, strategies: Dict[str, TradingStrategy], 
                                metrics: BacktestMetrics) -> Dict:
        """Gera relatório completo do backtest"""
        report = {
            'config': {
                'initial_capital': self.config.initial_capital,
                'commission_rate': self.config.commission_rate,
                'slippage_rate': self.config.slippage_rate,
                'max_positions': self.config.max_positions,
                'position_sizing_method': self.config.position_sizing_method,
                'timeframe': self.config.timeframe
            },
            'metrics': {
                'total_return': f"{metrics.total_return:.2%}",
                'annualized_return': f"{metrics.annualized_return:.2%}",
                'volatility': f"{metrics.volatility:.2%}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                'win_rate': f"{metrics.win_rate:.2%}",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'total_trades': metrics.total_trades,
                'avg_trade_duration': f"{metrics.avg_trade_duration:.1f} minutes",
                'total_fees': f"${metrics.total_fees:.2f}",
                'var_95': f"{metrics.var_95:.2%}",
                'expected_shortfall': f"{metrics.expected_shortfall:.2%}"
            },
            'portfolio_history': self.portfolio_history,
            'trades': [self._trade_to_dict(trade) for trade in self.trades],
            'strategies_used': list(strategies.keys()),
            'final_portfolio_value': self.portfolio_value,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Converte trade para dicionário"""
        return {
            'id': trade.id,
            'symbol': trade.symbol,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'fees': trade.fees,
            'duration_minutes': trade.duration_minutes,
            'entry_timestamp': trade.entry_timestamp.isoformat(),
            'exit_timestamp': trade.exit_timestamp.isoformat(),
            'strategy_id': trade.strategy_id,
            'exit_reason': trade.exit_reason
        }
    
    def generate_performance_plots(self, results: Dict, save_path: Optional[str] = None):
        """Gera gráficos de performance"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Configuração do estilo
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Equity Curve
            portfolio_history = results['portfolio_history']
            timestamps = [pd.to_datetime(h['timestamp']) for h in portfolio_history]
            values = [h['portfolio_value'] for h in portfolio_history]
            
            axes[0, 0].plot(timestamps, values, linewidth=2, color='blue')
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Drawdown
            running_max = np.maximum.accumulate(values)
            drawdowns = [(val - max_val) / max_val for val, max_val in zip(values, running_max)]
            
            axes[0, 1].fill_between(timestamps, drawdowns, 0, alpha=0.3, color='red')
            axes[0, 1].plot(timestamps, drawdowns, color='red', linewidth=1)
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Monthly Returns Heatmap
            if len(self.daily_returns) > 30:
                monthly_returns = self._calculate_monthly_returns()
                if monthly_returns:
                    df_monthly = pd.DataFrame(monthly_returns)
                    if not df_monthly.empty:
                        pivot_table = df_monthly.pivot_table(values='return', index='year', columns='month')
                        sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', 
                                  center=0, ax=axes[1, 0])
                        axes[1, 0].set_title('Monthly Returns Heatmap')
            
            # 4. Trade Distribution
            if results['trades']:
                pnls = [trade['pnl'] for trade in results['trades']]
                axes[1, 1].hist(pnls, bins=20, alpha=0.7, color='green', edgecolor='black')
                axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].set_title('Trade PnL Distribution')
                axes[1, 1].set_xlabel('PnL ($)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn não disponível para gráficos")
        except Exception as e:
            self.logger.error(f"Erro gerando gráficos: {str(e)}")
    
    def _calculate_monthly_returns(self) -> List[Dict]:
        """Calcula retornos mensais"""
        if not self.portfolio_history:
            return []
        
        monthly_data = {}
        
        for i, record in enumerate(self.portfolio_history):
            timestamp = pd.to_datetime(record['timestamp'])
            year_month = (timestamp.year, timestamp.month)
            
            if year_month not in monthly_data:
                monthly_data[year_month] = {'start_value': record['portfolio_value'], 'end_value': record['portfolio_value']}
            else:
                monthly_data[year_month]['end_value'] = record['portfolio_value']
        
        monthly_returns = []
        for (year, month), data in monthly_data.items():
            if data['start_value'] > 0:
                monthly_return = (data['end_value'] - data['start_value']) / data['start_value']
                monthly_returns.append({
                    'year': year,
                    'month': month,
                    'return': monthly_return
                })
        
        return monthly_returns

def run_strategy_comparison(strategies: Dict[str, TradingStrategy], 
                          market_data: Dict[str, pd.DataFrame],
                          config: BacktestConfig) -> Dict:
    """
    Executa comparação entre múltiplas estratégias
    
    Args:
        strategies: Dicionário de estratégias para comparar
        market_data: Dados de mercado
        config: Configuração do backtest
        
    Returns:
        Resultados comparativos
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    # Executa backtest para cada estratégia individualmente
    for strategy_name, strategy in strategies.items():
        logger.info(f"Testando estratégia: {strategy_name}")
        
        backtest_config = copy.deepcopy(config)
        backtester = AdvancedBacktester(backtest_config)
        
        try:
            result = backtester.run_backtest({strategy_name: strategy}, market_data)
            results[strategy_name] = result
        except Exception as e:
            logger.error(f"Erro testando {strategy_name}: {str(e)}")
            results[strategy_name] = {'error': str(e)}
    
    # Gera comparação
    comparison = _generate_strategy_comparison(results)
    
    return {
        'individual_results': results,
        'comparison': comparison,
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat()
    }

def _generate_strategy_comparison(results: Dict) -> Dict:
    """Gera comparação entre estratégias"""
    comparison_metrics = [
        'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'profit_factor', 'total_trades'
    ]
    
    comparison = {}
    
    for metric in comparison_metrics:
        comparison[metric] = {}
        
        for strategy_name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                # Remove formatação (%, $, etc.) e converte para float
                metric_value = result['metrics'].get(metric, '0')
                if isinstance(metric_value, str):
                    # Remove caracteres de formatação
                    clean_value = metric_value.replace('%', '').replace('$', '').replace(',', '')
                    try:
                        if metric == 'total_trades':
                            comparison[metric][strategy_name] = int(clean_value)
                        else:
                            comparison[metric][strategy_name] = float(clean_value)
                    except ValueError:
                        comparison[metric][strategy_name] = 0
                else:
                    comparison[metric][strategy_name] = metric_value
    
    # Ranking das estratégias
    ranking_metrics = ['sharpe_ratio', 'total_return', 'profit_factor']
    strategy_scores = {}
    
    for strategy_name in results.keys():
        if 'error' not in results[strategy_name]:
            score = 0
            for metric in ranking_metrics:
                if metric in comparison and strategy_name in comparison[metric]:
                    # Normaliza e soma scores
                    values = list(comparison[metric].values())
                    if values and max(values) > min(values):
                        normalized = (comparison[metric][strategy_name] - min(values)) / (max(values) - min(values))
                        score += normalized
            
            strategy_scores[strategy_name] = score
    
    # Ordena por score
    ranking = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
    
    comparison['ranking'] = ranking
    
    return comparison