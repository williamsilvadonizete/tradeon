"""
Módulo de análise técnica para trading
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Optional, Tuple
from ..exchanges.order_manager import OrderManager, OrderConfig
from ..risk.stop_loss_manager import StopLossManager, StopLossConfig
from .base import TradingStrategy
import sys
import os
import logging
from src.config.settings import (
    RSI_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BB_PERIOD,
    BB_STD,
    MAX_POSITION_SIZE,
    MIN_VOLUME,
    MIN_VOLATILITY,
    MAX_SPREAD
)

# Adiciona o diretório raiz ao path para importar configurações
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TechnicalAnalysisStrategy(TradingStrategy):
    """Estratégia de análise técnica"""
    
    def __init__(self, order_manager: OrderManager, stop_loss_manager: StopLossManager):
        """
        Inicializa a estratégia de análise técnica.
        
        Args:
            order_manager: Gerenciador de ordens
            stop_loss_manager: Gerenciador de stop loss
        """
        self.order_manager = order_manager
        self.stop_loss_manager = stop_loss_manager
        self.logger = self._setup_logger()
        
        # Configurações de futuros
        self.leverage = 5  # Alavancagem 5x
        self.margin_mode = 'cross'  # Modo de margem cross
        
    def _setup_logger(self) -> logging.Logger:
        """
        Configura o logger da estratégia.
        """
        logger = logging.getLogger('technical_analysis')
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo
        fh = logging.FileHandler('logs/technical_analysis.log')
        fh.setLevel(logging.INFO)
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato do log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
        """
        Analisa condições gerais do mercado
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            Dict com condições de mercado
        """
        try:
            # Calcula indicadores de tendência
            ema_fast = ta.trend.ema_indicator(data['close'], EMA_FAST)
            ema_slow = ta.trend.ema_indicator(data['close'], EMA_SLOW)
            
            # Calcula indicadores de momentum
            rsi = ta.momentum.rsi(data['close'], RSI_PERIOD)
            macd = ta.trend.MACD(
                data['close'],
                MACD_FAST,
                MACD_SLOW,
                MACD_SIGNAL
            )
            
            # Calcula indicadores de volatilidade
            bb = ta.volatility.BollingerBands(
                data['close'],
                BB_PERIOD,
                BB_STD
            )
            
            # Calcula volume médio
            avg_volume = data['volume'].rolling(20).mean()
            
            # Calcula spread
            spread = (data['high'] - data['low']) / data['low']
            
            # Determina tendência
            trend = 'up' if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 'down'
            
            # Determina força da tendência
            trend_strength = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
            
            # Determina condições de mercado
            market_conditions = {
                'trend': trend,
                'trend_strength': trend_strength,
                'rsi': rsi.iloc[-1],
                'macd': {
                    'line': macd.macd().iloc[-1],
                    'signal': macd.macd_signal().iloc[-1],
                    'histogram': macd.macd_diff().iloc[-1]
                },
                'bollinger_bands': {
                    'upper': bb.bollinger_hband().iloc[-1],
                    'middle': bb.bollinger_mavg().iloc[-1],
                    'lower': bb.bollinger_lband().iloc[-1]
                },
                'volume': {
                    'current': data['volume'].iloc[-1],
                    'average': avg_volume.iloc[-1]
                },
                'volatility': {
                    'current': spread.iloc[-1],
                    'average': spread.rolling(20).mean().iloc[-1]
                }
            }
            
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
            
    def validate_trade_conditions(self, market_conditions: Dict) -> bool:
        """
        Valida condições para execução de trade
        
        Args:
            market_conditions: Condições de mercado
            
        Returns:
            bool indicando se condições são válidas
        """
        try:
            # Verifica volume
            if market_conditions['volume']['current'] < MIN_VOLUME:
                return False
                
            # Verifica volatilidade
            if market_conditions['volatility']['current'] < MIN_VOLATILITY:
                return False
                
            # Verifica spread
            if market_conditions['volatility']['current'] > MAX_SPREAD:
                return False
                
            # Verifica força da tendência
            if market_conditions['trend_strength'] < 0.001:  # 0.1%
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade conditions: {str(e)}")
            return False
            
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Gera sinal de trading baseado em análise técnica
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            Dict com sinal de trading ou None
        """
        try:
            # Analisa condições de mercado
            market_conditions = self.analyze_market_conditions(data)
            
            # Valida condições
            if not self.validate_trade_conditions(market_conditions):
                return None
                
            # Determina sinal
            signal = None
            confidence = 0.0
            
            # Condições para compra
            if (market_conditions['trend'] == 'up' and
                market_conditions['rsi'] < 70 and
                market_conditions['macd']['histogram'] > 0 and
                data['close'].iloc[-1] < market_conditions['bollinger_bands']['upper']):
                
                signal = 'buy'
                confidence = min(
                    market_conditions['trend_strength'] * 10,
                    (70 - market_conditions['rsi']) / 70,
                    market_conditions['macd']['histogram'] / market_conditions['bollinger_bands']['middle']
                )
                
            # Condições para venda
            elif (market_conditions['trend'] == 'down' and
                  market_conditions['rsi'] > 30 and
                  market_conditions['macd']['histogram'] < 0 and
                  data['close'].iloc[-1] > market_conditions['bollinger_bands']['lower']):
                
                signal = 'sell'
                confidence = min(
                    market_conditions['trend_strength'] * 10,
                    (market_conditions['rsi'] - 30) / 70,
                    -market_conditions['macd']['histogram'] / market_conditions['bollinger_bands']['middle']
                )
                
            if signal and confidence > 0.5:  # Mínimo 50% de confiança
                return {
                    'action': signal,
                    'confidence': confidence,
                    'price': data['close'].iloc[-1],
                    'market_conditions': market_conditions
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None
        
    def initialize_futures_settings(self, symbol: str) -> None:
        """
        Inicializa configurações de futuros para o par.
        
        Args:
            symbol: Par de trading
        """
        try:
            # Configura alavancagem
            self.order_manager.exchange.set_leverage(symbol, self.leverage)
            self.logger.info(f"Set leverage to {self.leverage}x for {symbol}")
            
            # Configura modo de margem
            self.order_manager.exchange.set_margin_mode(symbol, self.margin_mode)
            self.logger.info(f"Set margin mode to {self.margin_mode} for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error setting futures parameters: {str(e)}")
            raise
            
    def get_position_size(self, signal: Dict, current_price: float,
                         available_balance: float) -> float:
        """
        Calcula o tamanho da posição baseado no sinal e saldo disponível.
        
        Args:
            signal: Sinal de trading
            current_price: Preço atual
            available_balance: Saldo disponível
            
        Returns:
            Tamanho da posição
        """
        # Ajusta o tamanho da posição baseado na confiança
        base_size = available_balance * 0.1  # 10% do saldo como base
        position_size = base_size * signal['confidence']
        
        # Ajusta para alavancagem
        leveraged_size = position_size * self.leverage
        
        # Converte para quantidade do ativo
        quantity = leveraged_size / current_price
        
        # Garante quantidade mínima
        if quantity < 0.001:  # Mínimo da Bybit para BTC/USDT
            quantity = 0.001
            
        return quantity
        
    def execute_trade(self, symbol: str, signal: Dict,
                     data: pd.DataFrame) -> Optional[Dict]:
        """
        Executa uma operação de trading.
        
        Args:
            symbol: Par de trading
            signal: Sinal de trading
            data: DataFrame com dados e indicadores
            
        Returns:
            Dicionário com informações da operação
        """
        if signal['action'] == 'hold':
            return None
            
        try:
            self.logger.info(f"Executing trade for {symbol} with signal: {signal['action']}")
            
            # Inicializa configurações de futuros
            self.initialize_futures_settings(symbol)
            
            # Obtém preço atual
            current_price = data['close'].iloc[-1]
            self.logger.info(f"Current price for {symbol}: {current_price}")
            
            # Converte ação para lado da posição
            position_side = 'long' if signal['action'] == 'buy' else 'short'
            
            # Calcula stop loss
            stop_price = self.stop_loss_manager.calculate_stop_loss(
                symbol=symbol,
                position_side=position_side,
                entry_price=current_price,
                historical_data=data
            )
            self.logger.info(f"Calculated stop loss for {symbol}: {stop_price}")
            
            # Calcula take profit com base no risco
            risk_reward_ratio = 2.0  # Alvo de lucro é 2x o risco
            if position_side == 'long':
                risk = current_price - stop_price
                take_profit = current_price + (risk * risk_reward_ratio)
            else:
                risk = stop_price - current_price
                take_profit = current_price - (risk * risk_reward_ratio)
            self.logger.info(f"Calculated take profit for {symbol}: {take_profit}")
                
            # Configura a ordem
            position_size = self.get_position_size(signal, current_price, 1000)  # TODO: Obter saldo real
            self.logger.info(f"Calculated position size for {symbol}: {position_size}")
            
            # Verifica saldo disponível
            try:
                balance = self.order_manager.exchange.get_balance('USDT')
                if not balance or 'free' not in balance or balance['free'] is None:
                    self.logger.warning(f"Could not get valid balance for USDT")
                    return None
                    
                required_margin = (position_size * current_price) / self.leverage
                available_balance = float(balance['free'])
                
                if available_balance < required_margin:
                    self.logger.warning(
                        f"Insufficient balance for {symbol}. Required margin: {required_margin} USDT, "
                        f"Available: {available_balance} USDT"
                    )
                    return None
                    
            except Exception as balance_error:
                self.logger.error(f"Error checking balance: {str(balance_error)}")
                return None
            
            order_config = OrderConfig(
                symbol=symbol,
                side=signal['action'],
                order_type='market',
                amount=position_size,
                stop_loss=stop_price,
                take_profit=take_profit
            )
            
            # Executa a ordem
            self.logger.info(f"Creating order for {symbol} with config: {order_config}")
            
            try:
                orders = self.order_manager.create_order_with_stops(order_config)
                if not orders:
                    self.logger.error(f"Failed to create orders for {symbol}")
                    return None
                    
                main_order, stop_order, tp_order = orders
                
                result = {
                    'main_order': main_order,
                    'stop_loss': stop_order,
                    'take_profit': tp_order,
                    'signal': signal,
                    'stop_info': {
                        'stop_price': stop_price,
                        'position_side': position_side,
                        'risk_reward_ratio': risk_reward_ratio,
                        'leverage': self.leverage,
                        'margin_mode': self.margin_mode
                    }
                }
                
                self.logger.info(f"Trade executed successfully for {symbol}: {result}")
                return result
                
            except Exception as order_error:
                self.logger.error(f"Error creating order for {symbol}: {str(order_error)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, data: pd.DataFrame) -> dict:
        """
        Calcula e retorna os indicadores técnicos principais para o DataFrame fornecido.
        """
        return self.analyze_market_conditions(data) 