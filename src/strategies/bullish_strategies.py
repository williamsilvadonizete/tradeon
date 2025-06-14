"""
Bullish Market Trading Strategies

This module implements specialized trading strategies optimized for bullish market conditions.
These strategies focus on momentum trading, breakout detection, and trend following.

Author: Trading Bot System
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .base import TradingStrategy
from ..analysis.indicators import TechnicalAnalyzer
from ..analysis.market_regime import MarketRegime, MarketRegimeDetector
from ..risk.advanced_risk_manager import AdvancedRiskManager

class BullishSignalType(Enum):
    """Tipos de sinais bullish"""
    MOMENTUM_BREAKOUT = "momentum_breakout"
    TREND_CONTINUATION = "trend_continuation"
    PULLBACK_ENTRY = "pullback_entry"
    VOLUME_SURGE = "volume_surge"
    TECHNICAL_BREAKOUT = "technical_breakout"

@dataclass
class BullishSignal:
    """Sinal de trading bullish"""
    signal_type: BullishSignalType
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple targets
    timeframe: str
    timestamp: datetime
    reason: str
    indicators: Dict

class MomentumBreakoutStrategy(TradingStrategy):
    """
    Estratégia de breakout de momentum para mercados bullish.
    Detecta rompimentos de resistência com volume e momentum.
    """
    
    def __init__(self, config: Dict = None):
        # Inicialização sem exchange para compatibilidade
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        
        # Configurações específicas da estratégia
        self.config.update({
            'momentum_threshold': 0.02,  # 2% movimento mínimo
            'volume_surge_factor': 1.5,  # 50% acima da média
            'breakout_confirmation_periods': 3,  # Confirmação por 3 períodos
            'atr_multiplier': 2.0,  # Para stop loss
            'risk_reward_ratio': 3.0,  # 1:3 risk/reward
            'max_pullback_percent': 0.03,  # 3% pullback máximo
            'trend_strength_min': 0.6,  # Força mínima da tendência
            'rsi_max_entry': 75,  # RSI máximo para entrada
            'ema_alignment_periods': [8, 21, 50],  # EMAs para alinhamento
        })
        
        self.active_positions = {}
        self.support_resistance_levels = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores específicos para breakout de momentum"""
        df = data.copy()
        
        # Indicadores básicos
        df = self.technical_analyzer.calculate_all_indicators(df)
        
        # ATR para volatilidade
        df['atr'] = self._calculate_atr(df, 14)
        
        # EMAs múltiplas para alinhamento de tendência
        for period in self.config['ema_alignment_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volume médio
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum personalizado
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Higher highs and higher lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['higher_low'] = (df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2))
        
        # Força da tendência bullish
        df['bullish_strength'] = self._calculate_bullish_strength(df)
        
        # Support and Resistance
        df = self._identify_support_resistance(df)
        
        return df
    
    def get_position_size(self, signal, current_price, available_balance):
        """Calcula tamanho da posição baseado no sinal"""
        if not signal or signal.get('action') not in ['buy', 'sell']:
            return 0.0
        
        confidence = signal.get('confidence', 0.5)
        base_position_size = available_balance * 0.1  # 10% base
        
        # Ajusta baseado na confiança
        adjusted_size = base_position_size * confidence
        
        # Converte para quantidade
        quantity = adjusted_size / current_price
        
        return quantity
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Gera sinais de breakout de momentum"""
        if len(data) < 50:
            return None
            
        df = self.calculate_indicators(data)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Verifica regime de mercado
        regime_signal = self.regime_detector.detect_regime(df.tail(100))
        if regime_signal.regime != MarketRegime.BULLISH:
            return None
        
        # Detecta breakout de momentum
        momentum_signal = self._detect_momentum_breakout(df)
        if momentum_signal:
            return momentum_signal
        
        # Detecta continuação de tendência
        trend_signal = self._detect_trend_continuation(df)
        if trend_signal:
            return trend_signal
        
        # Detecta entrada em pullback
        pullback_signal = self._detect_pullback_entry(df)
        if pullback_signal:
            return pullback_signal
        
        return None
    
    def _detect_momentum_breakout(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta breakout de momentum com confirmação"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Condições para breakout
        conditions = [
            # Breakout de preço
            current['close'] > current['resistance'],
            current['close'] > prev['high'],
            
            # Momentum forte
            current['momentum_5'] > self.config['momentum_threshold'],
            current['momentum_10'] > self.config['momentum_threshold'] * 0.5,
            
            # Volume confirmando
            current['volume_ratio'] > self.config['volume_surge_factor'],
            
            # Alinhamento de EMAs (bullish)
            current['ema_8'] > current['ema_21'] > current['ema_50'],
            
            # RSI não sobrecomprado
            current['rsi'] < self.config['rsi_max_entry'],
            
            # Força bullish
            current['bullish_strength'] > self.config['trend_strength_min']
        ]
        
        if sum(conditions) >= 6:  # Pelo menos 6 de 8 condições
            entry_price = current['close']
            stop_loss = current['close'] - (current['atr'] * self.config['atr_multiplier'])
            
            # Múltiplos take profits
            risk = entry_price - stop_loss
            take_profits = [
                entry_price + (risk * 1.5),  # Conservative
                entry_price + (risk * 2.5),  # Moderate
                entry_price + (risk * 4.0)   # Aggressive
            ]
            
            confidence = sum(conditions) / len(conditions)
            strength = min(current['momentum_5'] / self.config['momentum_threshold'], 1.0)
            
            return {
                'action': 'buy',
                'signal_type': BullishSignalType.MOMENTUM_BREAKOUT,
                'confidence': confidence,
                'strength': strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profits,
                'reason': f"Momentum breakout: {current['momentum_5']:.3f}, Volume: {current['volume_ratio']:.2f}x",
                'indicators': {
                    'momentum_5': current['momentum_5'],
                    'volume_ratio': current['volume_ratio'],
                    'rsi': current['rsi'],
                    'bullish_strength': current['bullish_strength']
                }
            }
        
        return None
    
    def _detect_trend_continuation(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta oportunidades de continuação de tendência"""
        current = df.iloc[-1]
        
        # Verifica se está em tendência bullish consistente
        lookback = df.tail(10)
        
        conditions = [
            # Série de higher highs
            sum(lookback['higher_high']) >= 3,
            
            # Preço acima de EMAs
            current['close'] > current['ema_8'],
            current['close'] > current['ema_21'],
            
            # EMAs alinhadas
            current['ema_8'] > current['ema_21'] > current['ema_50'],
            
            # Momentum positivo
            current['momentum_10'] > 0,
            
            # Não sobrecomprado
            current['rsi'] < 80,
            
            # Volume adequado
            current['volume_ratio'] > 0.8
        ]
        
        if sum(conditions) >= 5:
            entry_price = current['close']
            stop_loss = min(current['ema_21'], current['close'] * (1 - self.config['max_pullback_percent']))
            
            risk = entry_price - stop_loss
            take_profits = [
                entry_price + (risk * 2.0),
                entry_price + (risk * 3.5),
                entry_price + (risk * 5.0)
            ]
            
            confidence = sum(conditions) / len(conditions)
            strength = current['bullish_strength']
            
            return {
                'action': 'buy',
                'signal_type': BullishSignalType.TREND_CONTINUATION,
                'confidence': confidence,
                'strength': strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profits,
                'reason': f"Trend continuation: Higher highs count: {sum(lookback['higher_high'])}",
                'indicators': {
                    'higher_highs': sum(lookback['higher_high']),
                    'momentum_10': current['momentum_10'],
                    'rsi': current['rsi']
                }
            }
        
        return None
    
    def _detect_pullback_entry(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta entradas em pullbacks dentro de tendência bullish"""
        current = df.iloc[-1]
        recent = df.tail(5)
        
        # Verifica se está em pullback
        peak_price = recent['high'].max()
        current_pullback = (peak_price - current['close']) / peak_price
        
        conditions = [
            # Pullback moderado
            0.01 < current_pullback < self.config['max_pullback_percent'],
            
            # Ainda em tendência bullish de longo prazo
            current['ema_50'] < current['close'],
            current['ema_21'] < current['ema_8'],
            
            # RSI saiu de sobrecomprado
            30 < current['rsi'] < 60,
            
            # Suporte próximo
            abs(current['close'] - current['support']) / current['close'] < 0.02,
            
            # Volume diminuindo no pullback (sinal saudável)
            current['volume_ratio'] < 1.2,
            
            # Momentum ainda positivo no longo prazo
            current['momentum_10'] > 0
        ]
        
        if sum(conditions) >= 5:
            entry_price = current['close']
            stop_loss = min(current['support'] * 0.98, current['close'] * (1 - self.config['max_pullback_percent']))
            
            risk = entry_price - stop_loss
            take_profits = [
                peak_price,  # Retorno ao pico
                peak_price * 1.02,  # Breakout modesto
                peak_price * 1.05   # Breakout forte
            ]
            
            confidence = sum(conditions) / len(conditions)
            strength = 1 - current_pullback  # Força baseada no tamanho do pullback
            
            return {
                'action': 'buy',
                'signal_type': BullishSignalType.PULLBACK_ENTRY,
                'confidence': confidence,
                'strength': strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profits,
                'reason': f"Pullback entry: {current_pullback:.2%} from peak, Support at {current['support']:.2f}",
                'indicators': {
                    'pullback_percent': current_pullback,
                    'support_distance': abs(current['close'] - current['support']) / current['close'],
                    'rsi': current['rsi']
                }
            }
        
        return None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def _calculate_bullish_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calcula força bullish combinada"""
        # Componentes da força bullish
        momentum_score = np.tanh(df['momentum_10'] * 10)  # Normaliza momentum
        volume_score = np.minimum(df['volume_ratio'], 3) / 3  # Volume normalizado
        trend_score = (df['ema_8'] > df['ema_21']).astype(int)
        rsi_score = np.maximum(0, (df['rsi'] - 30) / 40)  # RSI entre 30-70
        
        # Combina scores
        bullish_strength = (momentum_score * 0.3 + 
                           volume_score * 0.25 + 
                           trend_score * 0.25 + 
                           rsi_score * 0.2)
        
        return np.clip(bullish_strength, 0, 1)
    
    def _identify_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Identifica níveis de suporte e resistência"""
        df = df.copy()
        
        # Support: menor preço em window períodos
        df['support'] = df['low'].rolling(window, center=True).min()
        
        # Resistance: maior preço em window períodos
        df['resistance'] = df['high'].rolling(window, center=True).max()
        
        # Preenche valores NaN
        df['support'] = df['support'].ffill().bfill()
        df['resistance'] = df['resistance'].ffill().bfill()
        
        return df

class TrendFollowingBullStrategy(TradingStrategy):
    """
    Estratégia de seguimento de tendência otimizada para mercados bullish.
    Foca em entradas durante correções em tendências de alta.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        
        self.config.update({
            'trend_periods': [20, 50, 100],  # Períodos para análise de tendência
            'pullback_max': 0.05,  # Pullback máximo de 5%
            'momentum_threshold': 0.001,  # Momentum mínimo
            'volume_confirmation': 1.2,  # Volume 20% acima da média
            'macd_bullish_threshold': 0,
            'rsi_oversold': 40,  # RSI para entradas
            'rsi_overbought': 75,
            'ema_distance_max': 0.03,  # Distância máxima da EMA
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para trend following"""
        df = self.technical_analyzer.calculate_all_indicators(data)
        
        # Múltiplas EMAs para trend analysis
        for period in self.config['trend_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Distância das EMAs
        df['ema_distance_20'] = (df['close'] - df['ema_20']) / df['ema_20']
        df['ema_distance_50'] = (df['close'] - df['ema_50']) / df['ema_50']
        
        # Trend slope
        df['trend_slope_20'] = df['ema_20'].pct_change(5)
        df['trend_slope_50'] = df['ema_50'].pct_change(10)
        
        # Volume médio
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def get_position_size(self, signal, current_price, available_balance):
        """Calcula tamanho da posição para trend following"""
        if not signal or signal.get('action') not in ['buy', 'sell']:
            return 0.0
        
        confidence = signal.get('confidence', 0.5)
        strength = signal.get('strength', 0.5)
        
        # Trend following pode usar posições maiores
        base_position_size = available_balance * 0.12  # 12% base
        
        # Ajusta baseado na confiança e força da tendência
        adjusted_size = base_position_size * confidence * strength
        
        # Converte para quantidade
        quantity = adjusted_size / current_price
        
        return quantity
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Gera sinais de trend following"""
        if len(data) < 100:
            return None
            
        df = self.calculate_indicators(data)
        current = df.iloc[-1]
        
        # Verifica condições de tendência bullish
        if not self._is_bullish_trend(df):
            return None
        
        # Verifica se está em pullback para entrada
        if self._is_pullback_entry_opportunity(df):
            return self._create_pullback_signal(df)
        
        return None
    
    def _is_bullish_trend(self, df: pd.DataFrame) -> bool:
        """Verifica se está em tendência bullish"""
        current = df.iloc[-1]
        
        conditions = [
            # EMAs alinhadas
            current['ema_20'] > current['ema_50'] > current['ema_100'],
            
            # Slopes positivos
            current['trend_slope_20'] > 0,
            current['trend_slope_50'] > 0,
            
            # MACD bullish
            current['macd'] > current['macd_signal'],
            
            # Preço acima da EMA principal
            current['close'] > current['ema_20']
        ]
        
        return sum(conditions) >= 4
    
    def _is_pullback_entry_opportunity(self, df: pd.DataFrame) -> bool:
        """Verifica se é uma oportunidade de entrada em pullback"""
        current = df.iloc[-1]
        recent_high = df['high'].tail(10).max()
        pullback = (recent_high - current['close']) / recent_high
        
        conditions = [
            # Pullback moderado
            0.01 < pullback < self.config['pullback_max'],
            
            # RSI não sobrecomprado
            self.config['rsi_oversold'] < current['rsi'] < self.config['rsi_overbought'],
            
            # Próximo da EMA
            abs(current['ema_distance_20']) < self.config['ema_distance_max'],
            
            # MACD ainda bullish
            current['macd'] > self.config['macd_bullish_threshold'],
            
            # Volume adequado
            current['volume_ratio'] > 0.8
        ]
        
        return sum(conditions) >= 4
    
    def _create_pullback_signal(self, df: pd.DataFrame) -> Dict:
        """Cria sinal de entrada em pullback"""
        current = df.iloc[-1]
        
        entry_price = current['close']
        stop_loss = current['ema_50']  # Stop na EMA 50
        
        # Take profit baseado na resistência recente
        recent_high = df['high'].tail(10).max()
        risk = entry_price - stop_loss
        
        take_profits = [
            recent_high,  # Primeiro target: retorno ao high
            recent_high + (risk * 0.5),  # Segundo target
            recent_high + (risk * 1.0)   # Terceiro target
        ]
        
        confidence = min(current['rsi'] / 100, 0.9)
        strength = abs(current['ema_distance_20'])
        
        return {
            'action': 'buy',
            'signal_type': BullishSignalType.PULLBACK_ENTRY,
            'confidence': confidence,
            'strength': strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profits,
            'reason': f"Trend following pullback: RSI {current['rsi']:.1f}, EMA distance {current['ema_distance_20']:.2%}",
            'indicators': {
                'rsi': current['rsi'],
                'ema_distance_20': current['ema_distance_20'],
                'macd': current['macd'],
                'volume_ratio': current['volume_ratio']
            }
        }

class VolumeSurgeStrategy(TradingStrategy):
    """
    Estratégia baseada em surtos de volume em mercados bullish.
    Detecta movimentos com volume anômalo que podem indicar continuação.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        
        self.config.update({
            'volume_surge_threshold': 2.0,  # Volume 2x acima da média
            'price_movement_min': 0.015,  # Movimento mínimo de 1.5%
            'confirmation_periods': 2,  # Períodos para confirmação
            'volume_lookback': 20,  # Períodos para média de volume
            'momentum_min': 0.01,  # Momentum mínimo
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para análise de volume"""
        df = self.technical_analyzer.calculate_all_indicators(data)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(self.config['volume_lookback']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_surge'] = df['volume_ratio'] > self.config['volume_surge_threshold']
        
        # Price momentum com volume
        df['price_change'] = df['close'].pct_change()
        df['volume_weighted_momentum'] = df['price_change'] * df['volume_ratio']
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        return df
    
    def get_position_size(self, signal, current_price, available_balance):
        """Calcula tamanho da posição para volume surge"""
        if not signal or signal.get('action') not in ['buy', 'sell']:
            return 0.0
        
        confidence = signal.get('confidence', 0.5)
        strength = signal.get('strength', 0.5)
        
        # Volume surge strategy - posições menores mas rápidas
        base_position_size = available_balance * 0.08  # 8% base
        
        # Ajusta baseado na força do volume surge
        adjusted_size = base_position_size * confidence * (1 + strength)
        
        # Converte para quantidade
        quantity = adjusted_size / current_price
        
        return quantity
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Gera sinais baseados em surtos de volume"""
        if len(data) < 50:
            return None
            
        df = self.calculate_indicators(data)
        current = df.iloc[-1]
        
        if self._detect_volume_surge_signal(df):
            return self._create_volume_surge_signal(df)
        
        return None
    
    def _detect_volume_surge_signal(self, df: pd.DataFrame) -> bool:
        """Detecta sinal de surto de volume"""
        current = df.iloc[-1]
        recent = df.tail(3)
        
        conditions = [
            # Volume surge atual
            current['volume_surge'],
            
            # Movimento de preço positivo
            current['price_change'] > self.config['price_movement_min'],
            
            # Momentum positivo
            current['volume_weighted_momentum'] > self.config['momentum_min'],
            
            # Preço acima do VWAP
            current['close'] > current['vwap'],
            
            # RSI não sobrecomprado
            current['rsi'] < 85,
            
            # Confirmação em períodos recentes
            sum(recent['volume_surge']) >= self.config['confirmation_periods']
        ]
        
        return sum(conditions) >= 5
    
    def _create_volume_surge_signal(self, df: pd.DataFrame) -> Dict:
        """Cria sinal de volume surge"""
        current = df.iloc[-1]
        
        entry_price = current['close']
        stop_loss = current['vwap'] * 0.98  # Stop abaixo do VWAP
        
        # Take profit baseado no momentum
        risk = entry_price - stop_loss
        momentum_multiplier = min(current['volume_ratio'], 5) / 2
        
        take_profits = [
            entry_price + (risk * 1.5 * momentum_multiplier),
            entry_price + (risk * 2.5 * momentum_multiplier),
            entry_price + (risk * 4.0 * momentum_multiplier)
        ]
        
        confidence = min(current['volume_ratio'] / self.config['volume_surge_threshold'], 1.0)
        strength = current['volume_weighted_momentum']
        
        return {
            'action': 'buy',
            'signal_type': BullishSignalType.VOLUME_SURGE,
            'confidence': confidence,
            'strength': strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profits,
            'reason': f"Volume surge: {current['volume_ratio']:.1f}x average, Price up {current['price_change']:.2%}",
            'indicators': {
                'volume_ratio': current['volume_ratio'],
                'price_change': current['price_change'],
                'volume_weighted_momentum': current['volume_weighted_momentum'],
                'vwap': current['vwap']
            }
        }

class BullishStrategyManager:
    """
    Gerenciador de estratégias bullish que combina múltiplas abordagens
    e seleciona a melhor estratégia baseada nas condições de mercado.
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Inicializa estratégias
        self.strategies = {
            'momentum_breakout': MomentumBreakoutStrategy(config),
            'trend_following': TrendFollowingBullStrategy(config),
            'volume_surge': VolumeSurgeStrategy(config)
        }
        
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdvancedRiskManager(config)
        
        # Performance tracking
        self.strategy_performance = {name: {'signals': 0, 'wins': 0, 'total_pnl': 0.0} 
                                   for name in self.strategies.keys()}
    
    def get_best_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Obtém o melhor sinal de todas as estratégias bullish.
        
        Args:
            data: DataFrame com dados de preço
            
        Returns:
            Melhor sinal disponível ou None
        """
        # Verifica se o regime é adequado
        regime_signal = self.regime_detector.detect_regime(data.tail(100))
        if regime_signal.regime not in [MarketRegime.BULLISH, MarketRegime.SIDEWAYS]:
            return None
        
        # Coleta sinais de todas as estratégias
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(data)
                if signal and signal.get('confidence', 0) > 0.5:
                    signals[name] = signal
            except Exception as e:
                self.logger.error(f"Error generating signal for {name}: {str(e)}")
        
        if not signals:
            return None
        
        # Seleciona o melhor sinal baseado em score combinado
        best_signal = None
        best_score = 0
        
        for name, signal in signals.items():
            # Calcula score baseado em confiança, força e performance histórica
            strategy_perf = self.strategy_performance[name]
            win_rate = strategy_perf['wins'] / max(strategy_perf['signals'], 1)
            
            score = (signal['confidence'] * 0.4 + 
                    signal['strength'] * 0.3 + 
                    win_rate * 0.2 + 
                    min(strategy_perf['total_pnl'] / 1000, 0.1) * 0.1)
            
            if score > best_score:
                best_score = score
                best_signal = signal.copy()
                best_signal['strategy'] = name
                best_signal['score'] = score
        
        return best_signal
    
    def update_strategy_performance(self, strategy_name: str, pnl: float):
        """Atualiza performance da estratégia"""
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name]['signals'] += 1
            self.strategy_performance[strategy_name]['total_pnl'] += pnl
            if pnl > 0:
                self.strategy_performance[strategy_name]['wins'] += 1
    
    def get_strategy_stats(self) -> Dict:
        """Retorna estatísticas das estratégias"""
        stats = {}
        for name, perf in self.strategy_performance.items():
            signals = perf['signals']
            win_rate = perf['wins'] / max(signals, 1)
            avg_pnl = perf['total_pnl'] / max(signals, 1)
            
            stats[name] = {
                'total_signals': signals,
                'win_rate': win_rate,
                'total_pnl': perf['total_pnl'],
                'average_pnl': avg_pnl
            }
        
        return stats