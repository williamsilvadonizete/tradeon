"""
Bearish Market Trading Strategies

This module implements specialized trading strategies optimized for bearish market conditions.
These strategies focus on short-selling, breakdown detection, and inverse momentum trading.

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

class BearishSignalType(Enum):
    """Tipos de sinais bearish"""
    BREAKDOWN = "breakdown"
    TREND_REVERSAL = "trend_reversal"
    DEAD_CAT_BOUNCE = "dead_cat_bounce"
    VOLUME_DUMP = "volume_dump"
    TECHNICAL_BREAKDOWN = "technical_breakdown"
    MOMENTUM_FADE = "momentum_fade"

@dataclass
class BearishSignal:
    """Sinal de trading bearish"""
    signal_type: BearishSignalType
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple targets
    timeframe: str
    timestamp: datetime
    reason: str
    indicators: Dict

class BreakdownStrategy(TradingStrategy):
    """
    Estratégia de breakdown para mercados bearish.
    Detecta rompimentos de suporte com volume e momentum negativo.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        
        # Configurações específicas da estratégia
        self.config.update({
            'breakdown_threshold': -0.02,  # 2% movimento mínimo para baixo
            'volume_surge_factor': 1.5,  # 50% acima da média
            'breakdown_confirmation_periods': 3,  # Confirmação por 3 períodos
            'atr_multiplier': 2.0,  # Para stop loss
            'risk_reward_ratio': 3.0,  # 1:3 risk/reward
            'max_bounce_percent': 0.03,  # 3% bounce máximo
            'trend_strength_min': 0.6,  # Força mínima da tendência
            'rsi_min_entry': 25,  # RSI mínimo para entrada
            'ema_alignment_periods': [8, 21, 50],  # EMAs para alinhamento
            'support_buffer': 0.005,  # Buffer de 0.5% para suporte
        })
        
        self.active_positions = {}
        self.support_resistance_levels = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores específicos para breakdown"""
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
        
        # Momentum personalizado (negativo para bearish)
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Lower highs and lower lows
        df['lower_high'] = (df['high'] < df['high'].shift(1)) & (df['high'].shift(1) < df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        
        # Força da tendência bearish
        df['bearish_strength'] = self._calculate_bearish_strength(df)
        
        # Support and Resistance
        df = self._identify_support_resistance(df)
        
        return df
    
    def get_position_size(self, signal, current_price, available_balance):
        """Calcula tamanho da posição baseado no sinal bearish"""
        if not signal or signal.get('action') not in ['buy', 'sell']:
            return 0.0
        
        confidence = signal.get('confidence', 0.5)
        strength = signal.get('strength', 0.5)
        
        # Bearish strategy - posições menores devido ao risco
        base_position_size = available_balance * 0.08  # 8% base
        
        # Ajusta baseado na confiança e força
        adjusted_size = base_position_size * confidence * strength
        
        # Converte para quantidade
        quantity = adjusted_size / current_price
        
        return quantity
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Gera sinais de breakdown"""
        if len(data) < 50:
            return None
            
        df = self.calculate_indicators(data)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Verifica regime de mercado
        regime_signal = self.regime_detector.detect_regime(df.tail(100))
        if regime_signal.regime not in [MarketRegime.BEARISH, MarketRegime.VOLATILE]:
            return None
        
        # Detecta breakdown de suporte
        breakdown_signal = self._detect_support_breakdown(df)
        if breakdown_signal:
            return breakdown_signal
        
        # Detecta continuação de tendência bearish
        trend_signal = self._detect_bearish_trend_continuation(df)
        if trend_signal:
            return trend_signal
        
        # Detecta dead cat bounce para shortar
        bounce_signal = self._detect_dead_cat_bounce(df)
        if bounce_signal:
            return bounce_signal
        
        return None
    
    def _detect_support_breakdown(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta breakdown de suporte com confirmação"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Condições para breakdown
        conditions = [
            # Breakdown de preço abaixo do suporte
            current['close'] < current['support'] * (1 - self.config['support_buffer']),
            current['close'] < prev['low'],
            
            # Momentum negativo forte
            current['momentum_5'] < self.config['breakdown_threshold'],
            current['momentum_10'] < self.config['breakdown_threshold'] * 0.5,
            
            # Volume confirmando
            current['volume_ratio'] > self.config['volume_surge_factor'],
            
            # Alinhamento de EMAs (bearish)
            current['ema_8'] < current['ema_21'] < current['ema_50'],
            
            # RSI não oversold (ainda tem espaço para cair)
            current['rsi'] > self.config['rsi_min_entry'],
            
            # Força bearish
            current['bearish_strength'] > self.config['trend_strength_min']
        ]
        
        if sum(conditions) >= 6:  # Pelo menos 6 de 8 condições
            entry_price = current['close']
            stop_loss = current['close'] + (current['atr'] * self.config['atr_multiplier'])
            
            # Múltiplos take profits
            risk = stop_loss - entry_price
            take_profits = [
                entry_price - (risk * 1.5),  # Conservative
                entry_price - (risk * 2.5),  # Moderate
                entry_price - (risk * 4.0)   # Aggressive
            ]
            
            confidence = sum(conditions) / len(conditions)
            strength = min(abs(current['momentum_5']) / abs(self.config['breakdown_threshold']), 1.0)
            
            return {
                'action': 'sell',
                'signal_type': BearishSignalType.BREAKDOWN,
                'confidence': confidence,
                'strength': strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profits,
                'reason': f"Support breakdown: {current['momentum_5']:.3f}, Volume: {current['volume_ratio']:.2f}x",
                'indicators': {
                    'momentum_5': current['momentum_5'],
                    'volume_ratio': current['volume_ratio'],
                    'rsi': current['rsi'],
                    'bearish_strength': current['bearish_strength'],
                    'support_level': current['support']
                }
            }
        
        return None
    
    def _detect_bearish_trend_continuation(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta oportunidades de continuação de tendência bearish"""
        current = df.iloc[-1]
        
        # Verifica se está em tendência bearish consistente
        lookback = df.tail(10)
        
        conditions = [
            # Série de lower lows
            sum(lookback['lower_low']) >= 3,
            
            # Preço abaixo de EMAs
            current['close'] < current['ema_8'],
            current['close'] < current['ema_21'],
            
            # EMAs alinhadas bearish
            current['ema_8'] < current['ema_21'] < current['ema_50'],
            
            # Momentum negativo
            current['momentum_10'] < 0,
            
            # Não oversold
            current['rsi'] > 20,
            
            # Volume adequado
            current['volume_ratio'] > 0.8
        ]
        
        if sum(conditions) >= 5:
            entry_price = current['close']
            stop_loss = max(current['ema_21'], current['close'] * (1 + self.config['max_bounce_percent']))
            
            risk = stop_loss - entry_price
            take_profits = [
                entry_price - (risk * 2.0),
                entry_price - (risk * 3.5),
                entry_price - (risk * 5.0)
            ]
            
            confidence = sum(conditions) / len(conditions)
            strength = current['bearish_strength']
            
            return {
                'action': 'sell',
                'signal_type': BearishSignalType.TREND_REVERSAL,
                'confidence': confidence,
                'strength': strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profits,
                'reason': f"Bearish trend continuation: Lower lows count: {sum(lookback['lower_low'])}",
                'indicators': {
                    'lower_lows': sum(lookback['lower_low']),
                    'momentum_10': current['momentum_10'],
                    'rsi': current['rsi']
                }
            }
        
        return None
    
    def _detect_dead_cat_bounce(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta dead cat bounce para shortar"""
        current = df.iloc[-1]
        recent = df.tail(5)
        
        # Verifica se está em bounce após queda
        low_price = recent['low'].min()
        current_bounce = (current['close'] - low_price) / low_price
        
        conditions = [
            # Bounce moderado após queda
            0.01 < current_bounce < self.config['max_bounce_percent'],
            
            # Ainda em tendência bearish de longo prazo
            current['ema_50'] > current['close'],
            current['ema_21'] > current['ema_8'],
            
            # RSI saiu de oversold mas não muito alto
            40 < current['rsi'] < 70,
            
            # Resistência próxima
            abs(current['close'] - current['resistance']) / current['close'] < 0.02,
            
            # Volume diminuindo no bounce (fraqueza)
            current['volume_ratio'] < 1.2,
            
            # Momentum ainda negativo no longo prazo
            current['momentum_10'] < 0
        ]
        
        if sum(conditions) >= 5:
            entry_price = current['close']
            stop_loss = min(current['resistance'] * 1.02, current['close'] * (1 + self.config['max_bounce_percent']))
            
            risk = stop_loss - entry_price
            take_profits = [
                low_price,  # Retorno ao low
                low_price * 0.98,  # Breakdown modesto
                low_price * 0.95   # Breakdown forte
            ]
            
            confidence = sum(conditions) / len(conditions)
            strength = 1 - current_bounce  # Força baseada no tamanho do bounce
            
            return {
                'action': 'sell',
                'signal_type': BearishSignalType.DEAD_CAT_BOUNCE,
                'confidence': confidence,
                'strength': strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profits,
                'reason': f"Dead cat bounce: {current_bounce:.2%} from low, Resistance at {current['resistance']:.2f}",
                'indicators': {
                    'bounce_percent': current_bounce,
                    'resistance_distance': abs(current['close'] - current['resistance']) / current['close'],
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
    
    def _calculate_bearish_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calcula força bearish combinada"""
        # Componentes da força bearish
        momentum_score = np.tanh(abs(df['momentum_10']) * 10) * np.sign(-df['momentum_10'])  # Negativo para bearish
        volume_score = np.minimum(df['volume_ratio'], 3) / 3
        trend_score = (df['ema_8'] < df['ema_21']).astype(int)
        rsi_score = np.maximum(0, (70 - df['rsi']) / 40)  # RSI entre 30-70, inverted for bearish
        
        # Combina scores
        bearish_strength = (abs(momentum_score) * 0.3 + 
                           volume_score * 0.25 + 
                           trend_score * 0.25 + 
                           rsi_score * 0.2)
        
        return np.clip(bearish_strength, 0, 1)
    
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

class TrendReversalStrategy(TradingStrategy):
    """
    Estratégia de reversão de tendência para detectar tops e iniciar shorts.
    Foca em sinais de exaustão de tendência bullish.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        
        self.config.update({
            'divergence_periods': 10,  # Períodos para detectar divergência
            'rsi_overbought': 75,  # RSI sobrecomprado
            'volume_decline_threshold': 0.7,  # Volume 30% abaixo da média
            'momentum_decline_periods': 5,  # Períodos de declínio do momentum
            'macd_bearish_cross': True,  # Requer cruzamento bearish do MACD
            'price_exhaustion_factor': 0.02,  # Fator de exaustão de preço
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para reversão de tendência"""
        df = self.technical_analyzer.calculate_all_indicators(data)
        
        # ATR para volatilidade
        df['atr'] = self._calculate_atr(df, 14)
        
        # Momentum básico primeiro
        df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        
        # Divergências RSI vs preço
        df['rsi_divergence'] = self._calculate_rsi_divergence(df)
        
        # Momentum de volume
        df['volume_momentum'] = df['volume'].pct_change(5)
        
        # Declining momentum
        df['momentum_declining'] = (df['momentum_5'] < df['momentum_5'].shift(1))
        
        # Volume médio
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price exhaustion pattern
        df['price_exhaustion'] = self._detect_price_exhaustion(df)
        
        return df
    
    def get_position_size(self, signal, current_price, available_balance):
        """Calcula tamanho da posição para reversão de tendência"""
        if not signal or signal.get('action') not in ['buy', 'sell']:
            return 0.0
        
        confidence = signal.get('confidence', 0.5)
        rsi = signal.get('indicators', {}).get('rsi', 50)
        
        # Trend reversal - ajusta baseado no RSI
        base_position_size = available_balance * 0.1  # 10% base
        
        # Maior posição quando RSI muito sobrecomprado
        rsi_multiplier = min(rsi / 100, 0.9) if rsi > 70 else 0.5
        adjusted_size = base_position_size * confidence * rsi_multiplier
        
        # Converte para quantidade
        quantity = adjusted_size / current_price
        
        return quantity
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Gera sinais de reversão de tendência"""
        if len(data) < 100:
            return None
            
        df = self.calculate_indicators(data)
        current = df.iloc[-1]
        
        # Detecta sinais de reversão
        if self._detect_trend_reversal_signal(df):
            return self._create_reversal_signal(df)
        
        return None
    
    def _detect_trend_reversal_signal(self, df: pd.DataFrame) -> bool:
        """Detecta sinal de reversão de tendência"""
        current = df.iloc[-1]
        recent = df.tail(self.config['divergence_periods'])
        
        conditions = [
            # RSI sobrecomprado
            current['rsi'] > self.config['rsi_overbought'],
            
            # Divergência RSI bearish
            current['rsi_divergence'] < 0,
            
            # MACD cruzamento bearish
            current['macd'] < current['macd_signal'],
            current['macd'] < recent['macd'].iloc[-2],  # MACD declining
            
            # Volume declining
            current['volume_ratio'] < self.config['volume_decline_threshold'],
            
            # Momentum declining
            sum(recent['momentum_declining']) >= 3,
            
            # Price exhaustion
            current['price_exhaustion'] > 0.5
        ]
        
        return sum(conditions) >= 5
    
    def _create_reversal_signal(self, df: pd.DataFrame) -> Dict:
        """Cria sinal de reversão"""
        current = df.iloc[-1]
        
        entry_price = current['close']
        
        # Stop loss baseado na resistência recente + ATR
        recent_high = df['high'].tail(10).max()
        stop_loss = recent_high + (current['atr'] * 1.5)
        
        # Take profit baseado em suportes
        risk = stop_loss - entry_price
        
        take_profits = [
            entry_price - (risk * 1.0),  # Quick profit
            entry_price - (risk * 2.0),  # Medium target
            entry_price - (risk * 3.5)   # Extended target
        ]
        
        confidence = min(current['rsi'] / 100, 0.95)
        strength = abs(current['rsi_divergence'])
        
        return {
            'action': 'sell',
            'signal_type': BearishSignalType.TREND_REVERSAL,
            'confidence': confidence,
            'strength': strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profits,
            'reason': f"Trend reversal: RSI {current['rsi']:.1f}, MACD bearish, Volume declining",
            'indicators': {
                'rsi': current['rsi'],
                'rsi_divergence': current['rsi_divergence'],
                'macd': current['macd'],
                'volume_ratio': current['volume_ratio'],
                'price_exhaustion': current['price_exhaustion']
            }
        }
    
    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calcula divergência entre RSI e preço"""
        periods = self.config['divergence_periods']
        
        # Price momentum
        price_momentum = df['close'].rolling(periods).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        )
        
        # RSI momentum
        rsi_momentum = df['rsi'].rolling(periods).apply(
            lambda x: x.iloc[-1] - x.iloc[0]
        )
        
        # Divergência: preço sobe mas RSI cai (bearish)
        divergence = np.where(
            (price_momentum > 0) & (rsi_momentum < 0),
            -abs(rsi_momentum / 100),  # Bearish divergence
            0
        )
        
        return pd.Series(divergence, index=df.index)
    
    def _detect_price_exhaustion(self, df: pd.DataFrame) -> pd.Series:
        """Detecta padrões de exaustão de preço"""
        # Doji patterns (open ≈ close)
        doji = abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1
        
        # Shooting star pattern
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        body = abs(df['close'] - df['open'])
        shooting_star = (upper_shadow > body * 2) & (df['close'] < df['open'])
        
        # Volume exhaustion
        volume_exhaustion = df['volume'] < df['volume'].rolling(10).mean() * 0.7
        
        # Combine patterns
        exhaustion_score = (doji.astype(int) * 0.4 + 
                           shooting_star.astype(int) * 0.6 + 
                           volume_exhaustion.astype(int) * 0.3)
        
        return np.clip(exhaustion_score, 0, 1)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()

class VolumeBreakdownStrategy(TradingStrategy):
    """
    Estratégia baseada em breakdowns com volume alto.
    Detecta vendas em pânico e liquidações forçadas.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        
        self.config.update({
            'volume_spike_threshold': 3.0,  # Volume 3x acima da média
            'price_drop_min': 0.02,  # Queda mínima de 2%
            'panic_selling_threshold': 0.05,  # Queda de 5% indica pânico
            'confirmation_periods': 2,  # Períodos para confirmação
            'volume_persistence': 3,  # Volume alto por 3 períodos
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para análise de volume breakdown"""
        df = self.technical_analyzer.calculate_all_indicators(data)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_spike'] = df['volume_ratio'] > self.config['volume_spike_threshold']
        
        # Price action com volume
        df['price_change'] = df['close'].pct_change()
        df['volume_weighted_price_change'] = df['price_change'] * df['volume_ratio']
        
        # Panic selling detection
        df['panic_selling'] = (df['price_change'] < -self.config['panic_selling_threshold']) & df['volume_spike']
        
        # Selling pressure
        df['selling_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def get_position_size(self, signal, current_price, available_balance):
        """Calcula tamanho da posição para volume breakdown"""
        if not signal or signal.get('action') not in ['buy', 'sell']:
            return 0.0
        
        confidence = signal.get('confidence', 0.5)
        volume_ratio = signal.get('indicators', {}).get('volume_ratio', 1.0)
        
        # Volume breakdown - posição maior com volume alto
        base_position_size = available_balance * 0.12  # 12% base
        
        # Ajusta baseado no volume surge
        volume_multiplier = min(volume_ratio / 2, 1.5)  # Max 1.5x
        adjusted_size = base_position_size * confidence * volume_multiplier
        
        # Converte para quantidade
        quantity = adjusted_size / current_price
        
        return quantity
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Gera sinais baseados em volume breakdown"""
        if len(data) < 50:
            return None
            
        df = self.calculate_indicators(data)
        
        if self._detect_volume_breakdown_signal(df):
            return self._create_volume_breakdown_signal(df)
        
        return None
    
    def _detect_volume_breakdown_signal(self, df: pd.DataFrame) -> bool:
        """Detecta sinal de volume breakdown"""
        current = df.iloc[-1]
        recent = df.tail(self.config['volume_persistence'])
        
        conditions = [
            # Volume spike atual
            current['volume_spike'],
            
            # Queda de preço significativa
            current['price_change'] < -self.config['price_drop_min'],
            
            # Volume alto persistente
            sum(recent['volume_spike']) >= self.config['confirmation_periods'],
            
            # Selling pressure alta
            current['selling_pressure'] < 0.3,  # Close near low
            
            # RSI não oversold (ainda pode cair)
            current['rsi'] > 30,
            
            # MACD bearish
            current['macd'] < current['macd_signal']
        ]
        
        return sum(conditions) >= 5
    
    def _create_volume_breakdown_signal(self, df: pd.DataFrame) -> Dict:
        """Cria sinal de volume breakdown"""
        current = df.iloc[-1]
        
        entry_price = current['close']
        
        # Stop loss baseado em resistência + volatilidade
        recent_resistance = df['high'].tail(5).max()
        stop_loss = recent_resistance * 1.02
        
        # Take profit baseado no momentum
        risk = stop_loss - entry_price
        volume_multiplier = min(current['volume_ratio'], 5) / 2
        
        take_profits = [
            entry_price - (risk * 1.5 * volume_multiplier),
            entry_price - (risk * 2.5 * volume_multiplier),
            entry_price - (risk * 4.0 * volume_multiplier)
        ]
        
        confidence = min(current['volume_ratio'] / self.config['volume_spike_threshold'], 1.0)
        strength = abs(current['volume_weighted_price_change'])
        
        return {
            'action': 'sell',
            'signal_type': BearishSignalType.VOLUME_DUMP,
            'confidence': confidence,
            'strength': strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profits,
            'reason': f"Volume breakdown: {current['volume_ratio']:.1f}x average, Price down {current['price_change']:.2%}",
            'indicators': {
                'volume_ratio': current['volume_ratio'],
                'price_change': current['price_change'],
                'selling_pressure': current['selling_pressure'],
                'rsi': current['rsi']
            }
        }

class BearishStrategyManager:
    """
    Gerenciador de estratégias bearish que combina múltiplas abordagens
    e seleciona a melhor estratégia baseada nas condições de mercado.
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Inicializa estratégias
        self.strategies = {
            'breakdown': BreakdownStrategy(config),
            'trend_reversal': TrendReversalStrategy(config),
            'volume_breakdown': VolumeBreakdownStrategy(config)
        }
        
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdvancedRiskManager(config)
        
        # Performance tracking
        self.strategy_performance = {name: {'signals': 0, 'wins': 0, 'total_pnl': 0.0} 
                                   for name in self.strategies.keys()}
    
    def get_best_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Obtém o melhor sinal de todas as estratégias bearish.
        
        Args:
            data: DataFrame com dados de preço
            
        Returns:
            Melhor sinal disponível ou None
        """
        # Verifica se o regime é adequado para estratégias bearish
        regime_signal = self.regime_detector.detect_regime(data.tail(100))
        if regime_signal.regime not in [MarketRegime.BEARISH, MarketRegime.VOLATILE]:
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