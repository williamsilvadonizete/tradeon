"""
Multi-Timeframe Analysis System

This module implements comprehensive multi-timeframe analysis for better trading decisions.
It analyzes market conditions across multiple timeframes to provide robust signals and
confirm trading opportunities.

Author: Trading Bot System
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import concurrent.futures
from threading import Lock

from .indicators import TechnicalAnalyzer
from .market_regime import MarketRegime, MarketRegimeDetector
from ..strategies.base import TradingStrategy

class TimeframePriority(Enum):
    """Prioridade dos timeframes para análise"""
    PRIMARY = "primary"      # Timeframe principal para execução
    SECONDARY = "secondary"  # Timeframe para confirmação
    TERTIARY = "tertiary"   # Timeframe para contexto

class TimeframeAlignment(Enum):
    """Alinhamento entre timeframes"""
    BULLISH = "bullish"     # Todos timeframes bullish
    BEARISH = "bearish"     # Todos timeframes bearish
    MIXED = "mixed"         # Timeframes conflitantes
    NEUTRAL = "neutral"     # Sem direção clara

@dataclass
class TimeframeSignal:
    """Sinal de um timeframe específico"""
    timeframe: str
    regime: MarketRegime
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float  # 0 to 1
    momentum: float  # -1 to 1
    volatility: float  # 0 to 1
    volume_profile: str  # 'high', 'normal', 'low'
    support_level: float
    resistance_level: float
    key_indicators: Dict
    confidence: float  # 0 to 1
    timestamp: datetime

@dataclass
class MultiTimeframeAnalysis:
    """Análise completa multi-timeframe"""
    primary_timeframe: str
    timeframe_signals: Dict[str, TimeframeSignal]
    overall_alignment: TimeframeAlignment
    primary_direction: str
    confidence_score: float
    risk_assessment: str
    entry_timeframe: str
    exit_timeframe: str
    recommendations: List[str]
    timestamp: datetime

class TimeframeManager:
    """
    Gerencia a configuração e hierarquia de timeframes
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Configuração padrão de timeframes
        self.timeframe_hierarchy = {
            '1m': {'priority': TimeframePriority.TERTIARY, 'weight': 0.1},
            '5m': {'priority': TimeframePriority.PRIMARY, 'weight': 0.3},
            '15m': {'priority': TimeframePriority.SECONDARY, 'weight': 0.25},
            '1h': {'priority': TimeframePriority.SECONDARY, 'weight': 0.25},
            '4h': {'priority': TimeframePriority.TERTIARY, 'weight': 0.1}
        }
        
        # Parâmetros de análise por timeframe
        self.timeframe_params = {
            '1m': {'lookback': 200, 'trend_periods': [10, 20], 'volatility_window': 20},
            '5m': {'lookback': 500, 'trend_periods': [20, 50], 'volatility_window': 50},
            '15m': {'lookback': 300, 'trend_periods': [20, 50], 'volatility_window': 30},
            '1h': {'lookback': 200, 'trend_periods': [20, 50, 100], 'volatility_window': 50},
            '4h': {'lookback': 150, 'trend_periods': [20, 50], 'volatility_window': 30}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_timeframe_weight(self, timeframe: str) -> float:
        """Retorna o peso do timeframe"""
        return self.timeframe_hierarchy.get(timeframe, {}).get('weight', 0.1)
    
    def get_primary_timeframe(self) -> str:
        """Retorna o timeframe primário"""
        for tf, config in self.timeframe_hierarchy.items():
            if config['priority'] == TimeframePriority.PRIMARY:
                return tf
        return '5m'  # Fallback
    
    def get_timeframe_params(self, timeframe: str) -> Dict:
        """Retorna parâmetros específicos do timeframe"""
        return self.timeframe_params.get(timeframe, self.timeframe_params['5m'])

class MultiTimeframeAnalyzer:
    """
    Analisador principal para múltiplos timeframes
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Componentes
        self.timeframe_manager = TimeframeManager(config)
        self.technical_analyzer = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        
        # Cache para análises
        self.analysis_cache = {}
        self.cache_lock = Lock()
        
        # Configurações
        self.analysis_config = {
            'cache_ttl_seconds': 300,  # 5 minutos
            'min_data_points': 50,
            'trend_confirmation_threshold': 0.6,
            'alignment_threshold': 0.7,
            'volatility_spike_threshold': 2.0,
            'volume_surge_threshold': 1.5
        }
        
        if config:
            self.analysis_config.update(config.get('multitimeframe', {}))
    
    def analyze_symbol(self, symbol: str, price_data: Dict[str, pd.DataFrame]) -> MultiTimeframeAnalysis:
        """
        Analisa um símbolo em múltiplos timeframes
        
        Args:
            symbol: Símbolo para análise
            price_data: Dados de preço por timeframe
            
        Returns:
            Análise completa multi-timeframe
        """
        try:
            # Verifica cache
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            with self.cache_lock:
                if cache_key in self.analysis_cache:
                    cached_analysis = self.analysis_cache[cache_key]
                    if (datetime.now() - cached_analysis.timestamp).total_seconds() < self.analysis_config['cache_ttl_seconds']:
                        return cached_analysis
            
            # Análise individual por timeframe
            timeframe_signals = {}
            
            # Usar ThreadPoolExecutor para análise paralela
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_timeframe = {
                    executor.submit(self._analyze_timeframe, symbol, timeframe, data): timeframe
                    for timeframe, data in price_data.items()
                    if len(data) >= self.analysis_config['min_data_points']
                }
                
                for future in concurrent.futures.as_completed(future_to_timeframe):
                    timeframe = future_to_timeframe[future]
                    try:
                        signal = future.get()
                        if signal:
                            timeframe_signals[timeframe] = signal
                    except Exception as e:
                        self.logger.error(f"Error analyzing {timeframe} for {symbol}: {str(e)}")
            
            # Análise combinada
            analysis = self._combine_timeframe_analysis(symbol, timeframe_signals)
            
            # Cache resultado
            with self.cache_lock:
                self.analysis_cache[cache_key] = analysis
                # Limita tamanho do cache
                if len(self.analysis_cache) > 100:
                    # Remove entradas mais antigas
                    oldest_key = min(self.analysis_cache.keys(), 
                                   key=lambda k: self.analysis_cache[k].timestamp)
                    del self.analysis_cache[oldest_key]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis for {symbol}: {str(e)}")
            return self._create_neutral_analysis(symbol)
    
    def _analyze_timeframe(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TimeframeSignal]:
        """Analisa um timeframe específico"""
        try:
            if len(data) < self.analysis_config['min_data_points']:
                return None
            
            # Calcula indicadores técnicos
            df = self.technical_analyzer.calculate_all_indicators(data.copy())
            
            # Obtém parâmetros específicos do timeframe
            params = self.timeframe_manager.get_timeframe_params(timeframe)
            
            # Detecta regime de mercado
            regime_signal = self.regime_detector.detect_regime(df.tail(params['lookback']))
            
            # Análise de tendência
            trend_analysis = self._analyze_trend(df, params['trend_periods'])
            
            # Análise de momentum
            momentum = self._calculate_momentum(df)
            
            # Análise de volatilidade
            volatility = self._analyze_volatility(df, params['volatility_window'])
            
            # Análise de volume
            volume_profile = self._analyze_volume_profile(df)
            
            # Suporte e resistência
            support, resistance = self._identify_key_levels(df)
            
            # Indicadores chave
            current = df.iloc[-1]
            key_indicators = {
                'rsi': current['rsi'],
                'macd': current['macd'],
                'macd_signal': current['macd_signal'],
                'ema_fast': current.get('ema_fast', 0),
                'ema_slow': current.get('ema_slow', 0),
                'bb_position': self._calculate_bb_position(current),
                'atr': self._calculate_current_atr(df)
            }
            
            # Calcula confiança baseada na consistência dos sinais
            confidence = self._calculate_timeframe_confidence(
                regime_signal, trend_analysis, momentum, volatility
            )
            
            return TimeframeSignal(
                timeframe=timeframe,
                regime=regime_signal.regime,
                trend_direction=trend_analysis['direction'],
                trend_strength=trend_analysis['strength'],
                momentum=momentum,
                volatility=volatility,
                volume_profile=volume_profile,
                support_level=support,
                resistance_level=resistance,
                key_indicators=key_indicators,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe}: {str(e)}")
            return None
    
    def _analyze_trend(self, df: pd.DataFrame, periods: List[int]) -> Dict:
        """Analisa tendência usando múltiplos períodos"""
        trend_scores = []
        current = df.iloc[-1]
        
        for period in periods:
            if len(df) > period:
                # EMA slope
                ema = df['close'].ewm(span=period).mean()
                slope = (ema.iloc[-1] - ema.iloc[-period//4]) / ema.iloc[-period//4]
                trend_scores.append(np.tanh(slope * 100))  # Normaliza slope
        
        if not trend_scores:
            return {'direction': 'sideways', 'strength': 0.0}
        
        avg_trend = np.mean(trend_scores)
        
        if avg_trend > 0.1:
            direction = 'up'
        elif avg_trend < -0.1:
            direction = 'down'
        else:
            direction = 'sideways'
        
        strength = min(abs(avg_trend), 1.0)
        
        return {'direction': direction, 'strength': strength}
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calcula momentum normalizado"""
        if len(df) < 20:
            return 0.0
        
        # Momentum de múltiplos períodos
        momentum_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
        momentum_10 = (df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11]
        momentum_20 = (df['close'].iloc[-1] - df['close'].iloc[-21]) / df['close'].iloc[-21]
        
        # Média ponderada
        combined_momentum = (momentum_5 * 0.5 + momentum_10 * 0.3 + momentum_20 * 0.2)
        
        # Normaliza usando tanh
        return np.tanh(combined_momentum * 20)
    
    def _analyze_volatility(self, df: pd.DataFrame, window: int) -> float:
        """Analisa volatilidade normalizada"""
        if len(df) < window:
            return 0.5
        
        # Volatilidade realizada
        returns = df['close'].pct_change().dropna()
        current_vol = returns.tail(window).std() * np.sqrt(252)  # Anualizada
        
        # Volatilidade histórica para normalização
        historical_vol = returns.std() * np.sqrt(252)
        
        if historical_vol == 0:
            return 0.5
        
        # Ratio normalizado
        vol_ratio = current_vol / historical_vol
        
        # Normaliza para 0-1
        return min(vol_ratio / self.analysis_config['volatility_spike_threshold'], 1.0)
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> str:
        """Analisa perfil de volume"""
        if len(df) < 20:
            return 'normal'
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        if avg_volume == 0:
            return 'normal'
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > self.analysis_config['volume_surge_threshold']:
            return 'high'
        elif volume_ratio < 0.7:
            return 'low'
        else:
            return 'normal'
    
    def _identify_key_levels(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Identifica níveis chave de suporte e resistência"""
        if len(df) < 20:
            current = df.iloc[-1]
            return current['low'], current['high']
        
        # Support: menor low dos últimos 20 períodos
        support = df['low'].tail(20).min()
        
        # Resistance: maior high dos últimos 20 períodos
        resistance = df['high'].tail(20).max()
        
        return support, resistance
    
    def _calculate_bb_position(self, current_row) -> float:
        """Calcula posição nas Bollinger Bands"""
        try:
            bb_high = current_row.get('bb_high', current_row['close'])
            bb_low = current_row.get('bb_low', current_row['close'])
            close = current_row['close']
            
            if bb_high == bb_low:
                return 0.5
            
            position = (close - bb_low) / (bb_high - bb_low)
            return np.clip(position, 0, 1)
        except:
            return 0.5
    
    def _calculate_current_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula ATR atual"""
        if len(df) < period:
            return 0.0
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.tail(period).mean()
    
    def _calculate_timeframe_confidence(self, regime_signal, trend_analysis, 
                                      momentum, volatility) -> float:
        """Calcula confiança do sinal do timeframe"""
        confidence_factors = [
            regime_signal.confidence,  # Confiança do regime
            trend_analysis['strength'],  # Força da tendência
            min(abs(momentum), 1.0),  # Strength do momentum
            1.0 - min(volatility, 1.0)  # Menor volatilidade = maior confiança
        ]
        
        return np.mean(confidence_factors)
    
    def _combine_timeframe_analysis(self, symbol: str, 
                                  timeframe_signals: Dict[str, TimeframeSignal]) -> MultiTimeframeAnalysis:
        """Combina análises de múltiplos timeframes"""
        if not timeframe_signals:
            return self._create_neutral_analysis(symbol)
        
        # Determina alinhamento geral
        alignment = self._determine_alignment(timeframe_signals)
        
        # Calcula direção primária com peso
        primary_direction = self._calculate_weighted_direction(timeframe_signals)
        
        # Calcula score de confiança geral
        confidence_score = self._calculate_overall_confidence(timeframe_signals)
        
        # Avaliação de risco
        risk_assessment = self._assess_risk(timeframe_signals, alignment)
        
        # Determina timeframes para entrada e saída
        entry_timeframe, exit_timeframe = self._determine_execution_timeframes(timeframe_signals)
        
        # Gera recomendações
        recommendations = self._generate_recommendations(timeframe_signals, alignment, confidence_score)
        
        # Timeframe primário
        primary_timeframe = self.timeframe_manager.get_primary_timeframe()
        
        return MultiTimeframeAnalysis(
            primary_timeframe=primary_timeframe,
            timeframe_signals=timeframe_signals,
            overall_alignment=alignment,
            primary_direction=primary_direction,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            entry_timeframe=entry_timeframe,
            exit_timeframe=exit_timeframe,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _determine_alignment(self, signals: Dict[str, TimeframeSignal]) -> TimeframeAlignment:
        """Determina alinhamento entre timeframes"""
        directions = [signal.trend_direction for signal in signals.values()]
        
        bullish_count = directions.count('up')
        bearish_count = directions.count('down')
        sideways_count = directions.count('sideways')
        
        total = len(directions)
        
        if bullish_count / total >= self.analysis_config['alignment_threshold']:
            return TimeframeAlignment.BULLISH
        elif bearish_count / total >= self.analysis_config['alignment_threshold']:
            return TimeframeAlignment.BEARISH
        elif sideways_count / total >= self.analysis_config['alignment_threshold']:
            return TimeframeAlignment.NEUTRAL
        else:
            return TimeframeAlignment.MIXED
    
    def _calculate_weighted_direction(self, signals: Dict[str, TimeframeSignal]) -> str:
        """Calcula direção com peso dos timeframes"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for timeframe, signal in signals.items():
            weight = self.timeframe_manager.get_timeframe_weight(timeframe)
            
            if signal.trend_direction == 'up':
                direction_score = signal.trend_strength
            elif signal.trend_direction == 'down':
                direction_score = -signal.trend_strength
            else:
                direction_score = 0
            
            weighted_score += direction_score * weight * signal.confidence
            total_weight += weight
        
        if total_weight == 0:
            return 'sideways'
        
        avg_score = weighted_score / total_weight
        
        if avg_score > 0.1:
            return 'up'
        elif avg_score < -0.1:
            return 'down'
        else:
            return 'sideways'
    
    def _calculate_overall_confidence(self, signals: Dict[str, TimeframeSignal]) -> float:
        """Calcula confiança geral ponderada"""
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for timeframe, signal in signals.items():
            weight = self.timeframe_manager.get_timeframe_weight(timeframe)
            weighted_confidence += signal.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _assess_risk(self, signals: Dict[str, TimeframeSignal], 
                    alignment: TimeframeAlignment) -> str:
        """Avalia risco geral baseado nos sinais"""
        avg_volatility = np.mean([signal.volatility for signal in signals.values()])
        
        # Fatores de risco
        risk_factors = []
        
        # Alinhamento de timeframes
        if alignment == TimeframeAlignment.MIXED:
            risk_factors.append("Timeframes conflitantes")
        
        # Volatilidade alta
        if avg_volatility > 0.7:
            risk_factors.append("Alta volatilidade")
        
        # Momentum inconsistente
        momentums = [signal.momentum for signal in signals.values()]
        momentum_std = np.std(momentums)
        if momentum_std > 0.5:
            risk_factors.append("Momentum inconsistente")
        
        # Classificação de risco
        if len(risk_factors) == 0:
            return "Baixo"
        elif len(risk_factors) <= 1:
            return "Moderado"
        else:
            return "Alto"
    
    def _determine_execution_timeframes(self, signals: Dict[str, TimeframeSignal]) -> Tuple[str, str]:
        """Determina timeframes para entrada e saída"""
        available_timeframes = list(signals.keys())
        
        # Entrada: timeframe com maior confiança entre os rápidos
        fast_timeframes = ['1m', '5m', '15m']
        entry_candidates = [tf for tf in fast_timeframes if tf in available_timeframes]
        
        if entry_candidates:
            entry_timeframe = max(entry_candidates, 
                                key=lambda tf: signals[tf].confidence)
        else:
            entry_timeframe = available_timeframes[0] if available_timeframes else '5m'
        
        # Saída: timeframe mais lento para trend following
        slow_timeframes = ['4h', '1h', '15m']
        exit_candidates = [tf for tf in slow_timeframes if tf in available_timeframes]
        
        if exit_candidates:
            exit_timeframe = exit_candidates[0]
        else:
            exit_timeframe = entry_timeframe
        
        return entry_timeframe, exit_timeframe
    
    def _generate_recommendations(self, signals: Dict[str, TimeframeSignal], 
                                alignment: TimeframeAlignment, 
                                confidence: float) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        # Recomendações baseadas no alinhamento
        if alignment == TimeframeAlignment.BULLISH:
            if confidence > 0.7:
                recommendations.append("Forte alinhamento bullish - considere posições longas")
            else:
                recommendations.append("Alinhamento bullish com confiança moderada")
        
        elif alignment == TimeframeAlignment.BEARISH:
            if confidence > 0.7:
                recommendations.append("Forte alinhamento bearish - considere posições curtas")
            else:
                recommendations.append("Alinhamento bearish com confiança moderada")
        
        elif alignment == TimeframeAlignment.MIXED:
            recommendations.append("Timeframes conflitantes - aguarde melhor alinhamento")
        
        # Recomendações baseadas em volatilidade
        avg_volatility = np.mean([signal.volatility for signal in signals.values()])
        if avg_volatility > 0.8:
            recommendations.append("Alta volatilidade - reduza tamanho de posição")
        
        # Recomendações baseadas em volume
        high_volume_timeframes = [tf for tf, signal in signals.items() 
                                if signal.volume_profile == 'high']
        if high_volume_timeframes:
            recommendations.append(f"Volume alto em {', '.join(high_volume_timeframes)}")
        
        return recommendations
    
    def _create_neutral_analysis(self, symbol: str) -> MultiTimeframeAnalysis:
        """Cria análise neutra quando não há dados suficientes"""
        return MultiTimeframeAnalysis(
            primary_timeframe='5m',
            timeframe_signals={},
            overall_alignment=TimeframeAlignment.NEUTRAL,
            primary_direction='sideways',
            confidence_score=0.0,
            risk_assessment="Dados insuficientes",
            entry_timeframe='5m',
            exit_timeframe='1h',
            recommendations=["Dados insuficientes para análise"],
            timestamp=datetime.now()
        )
    
    def get_signal_for_strategy(self, analysis: MultiTimeframeAnalysis, 
                              strategy_type: str = 'trend_following') -> Optional[Dict]:
        """
        Gera sinal específico para uma estratégia baseado na análise multi-timeframe
        
        Args:
            analysis: Análise multi-timeframe
            strategy_type: Tipo de estratégia ('trend_following', 'momentum', 'reversal')
            
        Returns:
            Sinal de trading ou None
        """
        if analysis.confidence_score < 0.5:
            return None
        
        primary_signal = analysis.timeframe_signals.get(analysis.primary_timeframe)
        if not primary_signal:
            return None
        
        # Sinal baseado no tipo de estratégia
        if strategy_type == 'trend_following':
            return self._create_trend_following_signal(analysis, primary_signal)
        elif strategy_type == 'momentum':
            return self._create_momentum_signal(analysis, primary_signal)
        elif strategy_type == 'reversal':
            return self._create_reversal_signal(analysis, primary_signal)
        
        return None
    
    def _create_trend_following_signal(self, analysis: MultiTimeframeAnalysis, 
                                     primary_signal: TimeframeSignal) -> Optional[Dict]:
        """Cria sinal de trend following"""
        if analysis.overall_alignment not in [TimeframeAlignment.BULLISH, TimeframeAlignment.BEARISH]:
            return None
        
        action = 'buy' if analysis.primary_direction == 'up' else 'sell'
        
        return {
            'action': action,
            'strategy_type': 'multi_timeframe_trend',
            'confidence': analysis.confidence_score,
            'strength': primary_signal.trend_strength,
            'entry_timeframe': analysis.entry_timeframe,
            'exit_timeframe': analysis.exit_timeframe,
            'stop_loss': primary_signal.support_level if action == 'buy' else primary_signal.resistance_level,
            'risk_assessment': analysis.risk_assessment,
            'timeframe_alignment': analysis.overall_alignment.value,
            'recommendations': analysis.recommendations
        }
    
    def _create_momentum_signal(self, analysis: MultiTimeframeAnalysis, 
                              primary_signal: TimeframeSignal) -> Optional[Dict]:
        """Cria sinal de momentum"""
        if abs(primary_signal.momentum) < 0.3:
            return None
        
        action = 'buy' if primary_signal.momentum > 0 else 'sell'
        
        return {
            'action': action,
            'strategy_type': 'multi_timeframe_momentum',
            'confidence': analysis.confidence_score * abs(primary_signal.momentum),
            'strength': abs(primary_signal.momentum),
            'momentum': primary_signal.momentum,
            'volatility': primary_signal.volatility,
            'volume_profile': primary_signal.volume_profile,
            'timeframe_alignment': analysis.overall_alignment.value
        }
    
    def _create_reversal_signal(self, analysis: MultiTimeframeAnalysis, 
                              primary_signal: TimeframeSignal) -> Optional[Dict]:
        """Cria sinal de reversão"""
        # Reversão requer timeframes conflitantes ou condições extremas
        if analysis.overall_alignment != TimeframeAlignment.MIXED:
            return None
        
        # Detecta condições de reversão
        reversal_conditions = []
        
        # RSI extremo
        rsi = primary_signal.key_indicators.get('rsi', 50)
        if rsi > 80 or rsi < 20:
            reversal_conditions.append('rsi_extreme')
        
        # Bollinger Bands extremas
        bb_position = primary_signal.key_indicators.get('bb_position', 0.5)
        if bb_position > 0.9 or bb_position < 0.1:
            reversal_conditions.append('bb_extreme')
        
        if len(reversal_conditions) < 1:
            return None
        
        # Direção da reversão (oposta ao momentum atual)
        action = 'sell' if primary_signal.momentum > 0 else 'buy'
        
        return {
            'action': action,
            'strategy_type': 'multi_timeframe_reversal',
            'confidence': analysis.confidence_score * 0.7,  # Menor confiança para reversões
            'strength': len(reversal_conditions) / 2,
            'reversal_conditions': reversal_conditions,
            'risk_assessment': 'Alto',  # Reversões são mais arriscadas
            'timeframe_alignment': analysis.overall_alignment.value
        }