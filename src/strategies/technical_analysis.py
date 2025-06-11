import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from ..exchanges.order_manager import OrderManager, OrderConfig
from ..risk.stop_loss_manager import StopLossManager, StopLossConfig
from .base import TradingStrategy
import sys
import os
import logging

# Adiciona o diretório raiz ao path para importar configurações
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    RSI_PERIOD, EMA_FAST, EMA_SLOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, MAX_POSITION_SIZE
)

class TechnicalAnalysisStrategy(TradingStrategy):
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
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos.
        
        Args:
            data: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com indicadores adicionados
        """
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # EMAs
        data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
        data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
        data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['std_20'] = data['close'].rolling(window=20).std()
        data['upper_band'] = data['sma_20'] + (data['std_20'] * 2)
        data['lower_band'] = data['sma_20'] - (data['std_20'] * 2)
        
        return data
        
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Gera sinal de trading baseado nos indicadores.
        
        Args:
            data: DataFrame com indicadores
            
        Returns:
            Dicionário com sinal e informações adicionais
        """
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2]
        
        # Inicializa sinal
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'reason': []
        }
        
        # Análise RSI
        if last_row['rsi'] < 30:
            signal['action'] = 'buy'
            signal['confidence'] += 0.3
            signal['reason'].append('RSI sobrevendido')
        elif last_row['rsi'] > 70:
            signal['action'] = 'sell'
            signal['confidence'] += 0.3
            signal['reason'].append('RSI sobrecomprado')
            
        # Análise EMAs
        if last_row['ema_9'] > last_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']:
            signal['action'] = 'buy'
            signal['confidence'] += 0.2
            signal['reason'].append('Cruzamento de EMAs para cima')
        elif last_row['ema_9'] < last_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']:
            signal['action'] = 'sell'
            signal['confidence'] += 0.2
            signal['reason'].append('Cruzamento de EMAs para baixo')
            
        # Análise MACD
        if last_row['macd'] > last_row['signal'] and prev_row['macd'] <= prev_row['signal']:
            signal['action'] = 'buy'
            signal['confidence'] += 0.2
            signal['reason'].append('Cruzamento MACD para cima')
        elif last_row['macd'] < last_row['signal'] and prev_row['macd'] >= prev_row['signal']:
            signal['action'] = 'sell'
            signal['confidence'] += 0.2
            signal['reason'].append('Cruzamento MACD para baixo')
            
        # Análise Bollinger Bands
        if last_row['close'] < last_row['lower_band']:
            signal['action'] = 'buy'
            signal['confidence'] += 0.3
            signal['reason'].append('Preço abaixo da banda inferior')
        elif last_row['close'] > last_row['upper_band']:
            signal['action'] = 'sell'
            signal['confidence'] += 0.3
            signal['reason'].append('Preço acima da banda superior')
            
        # Normaliza confiança
        signal['confidence'] = min(signal['confidence'], 1.0)
        
        return signal
        
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