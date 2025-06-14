import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from enum import Enum
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class MarketRegime(Enum):
    """Diferentes regimes de mercado"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class RegimeSignal:
    """Sinal de regime de mercado"""
    regime: MarketRegime
    confidence: float
    factors: List[str]
    volatility_level: str  # low, medium, high
    trend_strength: float

class MarketRegimeDetector:
    """
    Detector avançado de regime de mercado usando múltiplos indicadores
    e machine learning para classificação inteligente.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lookback_period = 50
        self.volatility_window = 20
        self.scaler = StandardScaler()
        self.model = None
        self._initialize_model()
        
        # Thresholds para classificação
        self.thresholds = {
            'trend_strength': 0.15,  # 15% movimento mínimo para tendência forte
            'volatility_high': 2.0,  # 2x ATR médio para alta volatilidade
            'sideways_range': 0.05   # 5% range para mercado lateral
        }
    
    def _initialize_model(self):
        """Inicializa ou carrega modelo de ML"""
        model_path = 'models/regime_classifier.joblib'
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.logger.info("Loaded trained regime classification model")
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}. Using rule-based approach.")
                self.model = None
        else:
            self.logger.info("No trained model found. Using rule-based regime detection.")
            self.model = None
    
    def detect_regime(self, data: pd.DataFrame) -> RegimeSignal:
        """
        Detecta o regime atual de mercado usando análise técnica e ML.
        
        Args:
            data: DataFrame com dados OHLCV e indicadores
            
        Returns:
            RegimeSignal com classificação e detalhes
        """
        try:
            if data is None or len(data) < self.lookback_period:
                return RegimeSignal(
                    regime=MarketRegime.SIDEWAYS,
                    confidence=0.5,
                    factors=["Insufficient data"],
                    volatility_level="medium",
                    trend_strength=0.0
                )
            
            # Calcula features para análise
            features = self._extract_regime_features(data)
            
            # Análise baseada em regras
            rule_based_regime = self._rule_based_classification(features, data)
            
            # Análise usando ML (se disponível)
            ml_regime = None
            if self.model is not None:
                ml_regime = self._ml_classification(features)
            
            # Combina resultados
            final_regime = self._combine_classifications(rule_based_regime, ml_regime)
            
            return final_regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return RegimeSignal(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.3,
                factors=["Error in analysis"],
                volatility_level="medium",
                trend_strength=0.0
            )
    
    def _extract_regime_features(self, data: pd.DataFrame) -> Dict:
        """Extrai features para classificação de regime"""
        try:
            current_price = data['close'].iloc[-1]
            
            # 1. Trend Analysis
            trend_features = self._calculate_trend_features(data)
            
            # 2. Volatility Analysis
            volatility_features = self._calculate_volatility_features(data)
            
            # 3. Momentum Analysis
            momentum_features = self._calculate_momentum_features(data)
            
            # 4. Volume Analysis
            volume_features = self._calculate_volume_features(data)
            
            # 5. Price Action Features
            price_action_features = self._calculate_price_action_features(data)
            
            # Combina todas as features
            features = {
                **trend_features,
                **volatility_features,
                **momentum_features,
                **volume_features,
                **price_action_features
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting regime features: {str(e)}")
            return {}
    
    def _calculate_trend_features(self, data: pd.DataFrame) -> Dict:
        """Calcula features de tendência"""
        current_price = data['close'].iloc[-1]
        
        # EMAs para diferentes períodos
        ema_5 = data['close'].ewm(span=5).mean().iloc[-1]
        ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = data['close'].ewm(span=50).mean().iloc[-1]
        
        # Price relative to EMAs
        price_vs_ema5 = (current_price - ema_5) / ema_5
        price_vs_ema20 = (current_price - ema_20) / ema_20
        price_vs_ema50 = (current_price - ema_50) / ema_50
        
        # EMA alignment
        ema_bullish_alignment = (ema_5 > ema_20 > ema_50)
        ema_bearish_alignment = (ema_5 < ema_20 < ema_50)
        
        # Trend strength over different periods
        trend_5d = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        trend_20d = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        trend_50d = (data['close'].iloc[-1] - data['close'].iloc[-50]) / data['close'].iloc[-50] if len(data) >= 50 else 0
        
        return {
            'price_vs_ema5': price_vs_ema5,
            'price_vs_ema20': price_vs_ema20,
            'price_vs_ema50': price_vs_ema50,
            'ema_bullish_alignment': int(ema_bullish_alignment),
            'ema_bearish_alignment': int(ema_bearish_alignment),
            'trend_5d': trend_5d,
            'trend_20d': trend_20d,
            'trend_50d': trend_50d,
            'trend_consistency': abs(trend_5d + trend_20d + trend_50d) / 3
        }
    
    def _calculate_volatility_features(self, data: pd.DataFrame) -> Dict:
        """Calcula features de volatilidade"""
        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(50).mean().iloc[-1]
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Realized volatility
        returns = data['close'].pct_change().dropna()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        current_vol = realized_vol.iloc[-1]
        avg_vol = realized_vol.rolling(50).mean().iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        # Bollinger Band position
        if 'bb_high' in data.columns and 'bb_low' in data.columns:
            bb_position = (data['close'].iloc[-1] - data['bb_low'].iloc[-1]) / (data['bb_high'].iloc[-1] - data['bb_low'].iloc[-1])
        else:
            bb_position = 0.5
        
        return {
            'atr_ratio': atr_ratio,
            'volatility_ratio': vol_ratio,
            'current_volatility': current_vol,
            'bb_position': bb_position,
            'volatility_regime': 'high' if vol_ratio > 1.5 else 'low' if vol_ratio < 0.7 else 'medium'
        }
    
    def _calculate_momentum_features(self, data: pd.DataFrame) -> Dict:
        """Calcula features de momentum"""
        # RSI
        rsi = data.get('rsi', pd.Series([50] * len(data))).iloc[-1]
        
        # MACD
        macd = data.get('macd', pd.Series([0] * len(data))).iloc[-1]
        macd_signal = data.get('macd_signal', pd.Series([0] * len(data))).iloc[-1]
        macd_histogram = macd - macd_signal
        
        # Rate of Change
        roc_5 = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        roc_20 = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        
        # Momentum consistency
        momentum_signals = [
            1 if rsi > 60 else -1 if rsi < 40 else 0,
            1 if macd > macd_signal else -1,
            1 if roc_5 > 0.02 else -1 if roc_5 < -0.02 else 0,
            1 if roc_20 > 0.05 else -1 if roc_20 < -0.05 else 0
        ]
        momentum_consistency = sum(momentum_signals) / len(momentum_signals)
        
        return {
            'rsi': rsi,
            'macd_histogram': macd_histogram,
            'roc_5d': roc_5,
            'roc_20d': roc_20,
            'momentum_consistency': momentum_consistency,
            'momentum_strength': abs(momentum_consistency)
        }
    
    def _calculate_volume_features(self, data: pd.DataFrame) -> Dict:
        """Calcula features de volume"""
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        volume_trend = (data['volume'].rolling(5).mean().iloc[-1] - 
                       data['volume'].rolling(20).mean().iloc[-1]) / data['volume'].rolling(20).mean().iloc[-1]
        
        # Price-Volume correlation
        price_returns = data['close'].pct_change().tail(20)
        volume_changes = data['volume'].pct_change().tail(20)
        pv_correlation = price_returns.corr(volume_changes) if len(price_returns) > 1 else 0
        
        return {
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'price_volume_correlation': pv_correlation,
            'volume_regime': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.7 else 'normal'
        }
    
    def _calculate_price_action_features(self, data: pd.DataFrame) -> Dict:
        """Calcula features de price action"""
        # Higher highs, lower lows analysis
        recent_highs = data['high'].tail(10)
        recent_lows = data['low'].tail(10)
        
        higher_highs = sum(recent_highs.iloc[i] > recent_highs.iloc[i-1] for i in range(1, len(recent_highs)))
        lower_lows = sum(recent_lows.iloc[i] < recent_lows.iloc[i-1] for i in range(1, len(recent_lows)))
        
        # Support/Resistance analysis
        current_price = data['close'].iloc[-1]
        recent_data = data.tail(50)
        resistance_level = recent_data['high'].max()
        support_level = recent_data['low'].min()
        
        distance_to_resistance = (resistance_level - current_price) / current_price
        distance_to_support = (current_price - support_level) / current_price
        
        # Range analysis
        range_size = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
        
        return {
            'higher_highs_count': higher_highs,
            'lower_lows_count': lower_lows,
            'distance_to_resistance': distance_to_resistance,
            'distance_to_support': distance_to_support,
            'range_size': range_size,
            'price_position_in_range': (current_price - support_level) / (resistance_level - support_level) if resistance_level != support_level else 0.5
        }
    
    def _rule_based_classification(self, features: Dict, data: pd.DataFrame) -> RegimeSignal:
        """Classificação baseada em regras técnicas"""
        try:
            factors = []
            confidence = 0.0
            
            # Análise de tendência
            trend_score = 0
            if features.get('trend_20d', 0) > 0.1:  # +10% em 20 dias
                trend_score += 2
                factors.append("Forte tendência de alta (20d)")
            elif features.get('trend_20d', 0) < -0.1:  # -10% em 20 dias
                trend_score -= 2
                factors.append("Forte tendência de baixa (20d)")
            
            # Análise de alinhamento de EMAs
            if features.get('ema_bullish_alignment', 0):
                trend_score += 1
                factors.append("Alinhamento bullish de EMAs")
            elif features.get('ema_bearish_alignment', 0):
                trend_score -= 1
                factors.append("Alinhamento bearish de EMAs")
            
            # Análise de momentum
            momentum_score = features.get('momentum_consistency', 0)
            if momentum_score > 0.5:
                trend_score += 1
                factors.append("Momentum positivo consistente")
            elif momentum_score < -0.5:
                trend_score -= 1
                factors.append("Momentum negativo consistente")
            
            # Análise de volatilidade
            vol_regime = features.get('volatility_regime', 'medium')
            atr_ratio = features.get('atr_ratio', 1.0)
            
            # Determinação do regime
            if abs(trend_score) <= 1 and features.get('range_size', 0) < 0.1:
                # Mercado lateral
                regime = MarketRegime.SIDEWAYS
                confidence = 0.7
                factors.append("Movimentação lateral com baixo range")
                
            elif vol_regime == 'high' and atr_ratio > 2.0:
                # Mercado volátil
                regime = MarketRegime.VOLATILE
                confidence = 0.8
                factors.append(f"Alta volatilidade (ATR {atr_ratio:.1f}x média)")
                
            elif trend_score >= 2:
                # Mercado bullish
                regime = MarketRegime.BULLISH
                confidence = min(0.9, 0.6 + (trend_score - 2) * 0.1)
                factors.append("Múltiplos fatores bullish confirmados")
                
            elif trend_score <= -2:
                # Mercado bearish
                regime = MarketRegime.BEARISH
                confidence = min(0.9, 0.6 + abs(trend_score + 2) * 0.1)
                factors.append("Múltiplos fatores bearish confirmados")
                
            else:
                # Regime incerto - default para sideways
                regime = MarketRegime.SIDEWAYS
                confidence = 0.5
                factors.append("Sinais mistos - regime incerto")
            
            # Determina nível de volatilidade
            if atr_ratio > 1.5:
                volatility_level = "high"
            elif atr_ratio < 0.7:
                volatility_level = "low"
            else:
                volatility_level = "medium"
            
            return RegimeSignal(
                regime=regime,
                confidence=confidence,
                factors=factors,
                volatility_level=volatility_level,
                trend_strength=abs(features.get('trend_20d', 0))
            )
            
        except Exception as e:
            self.logger.error(f"Error in rule-based classification: {str(e)}")
            return RegimeSignal(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.3,
                factors=["Error in rule-based analysis"],
                volatility_level="medium",
                trend_strength=0.0
            )
    
    def _ml_classification(self, features: Dict) -> RegimeSignal:
        """Classificação usando machine learning"""
        try:
            if self.model is None:
                return None
            
            # Prepara features para o modelo
            feature_vector = self._prepare_features_for_ml(features)
            
            # Faz predição
            probabilities = self.model.predict_proba([feature_vector])[0]
            predicted_class = self.model.predict([feature_vector])[0]
            
            # Mapeia classe para regime
            regime_mapping = {
                0: MarketRegime.BEARISH,
                1: MarketRegime.SIDEWAYS,
                2: MarketRegime.BULLISH,
                3: MarketRegime.VOLATILE
            }
            
            regime = regime_mapping.get(predicted_class, MarketRegime.SIDEWAYS)
            confidence = max(probabilities)
            
            return RegimeSignal(
                regime=regime,
                confidence=confidence,
                factors=["Machine Learning prediction"],
                volatility_level="medium",  # Will be overridden by rule-based
                trend_strength=0.0  # Will be overridden by rule-based
            )
            
        except Exception as e:
            self.logger.error(f"Error in ML classification: {str(e)}")
            return None
    
    def _prepare_features_for_ml(self, features: Dict) -> List[float]:
        """Prepara features para o modelo de ML"""
        # Lista de features esperadas pelo modelo (em ordem)
        feature_names = [
            'price_vs_ema20', 'trend_20d', 'atr_ratio', 'volatility_ratio',
            'rsi', 'momentum_consistency', 'volume_ratio', 'range_size'
        ]
        
        feature_vector = []
        for feature_name in feature_names:
            value = features.get(feature_name, 0.0)
            # Sanitiza valores para evitar problemas
            if pd.isna(value) or np.isinf(value):
                value = 0.0
            feature_vector.append(float(value))
        
        return feature_vector
    
    def _combine_classifications(self, rule_based: RegimeSignal, ml_based: RegimeSignal) -> RegimeSignal:
        """Combina classificações baseadas em regras e ML"""
        if ml_based is None:
            return rule_based
        
        # Se ambos concordam, aumenta a confiança
        if rule_based.regime == ml_based.regime:
            combined_confidence = min(0.95, (rule_based.confidence + ml_based.confidence) / 2 + 0.1)
            factors = rule_based.factors + ["ML confirmation"]
        else:
            # Se discordam, usa regras mas reduz confiança
            combined_confidence = rule_based.confidence * 0.8
            factors = rule_based.factors + [f"ML suggests {ml_based.regime.value}"]
        
        return RegimeSignal(
            regime=rule_based.regime,
            confidence=combined_confidence,
            factors=factors,
            volatility_level=rule_based.volatility_level,
            trend_strength=rule_based.trend_strength
        )
    
    def get_regime_strategies(self, regime: MarketRegime) -> Dict:
        """Retorna estratégias recomendadas para cada regime"""
        strategies = {
            MarketRegime.BULLISH: {
                'primary': 'momentum_following',
                'secondary': 'breakout_trading',
                'risk_level': 'medium_high',
                'position_sizing': 'aggressive',
                'stop_loss_type': 'trailing'
            },
            MarketRegime.BEARISH: {
                'primary': 'mean_reversion',
                'secondary': 'breakdown_trading',
                'risk_level': 'conservative',
                'position_sizing': 'defensive',
                'stop_loss_type': 'tight_fixed'
            },
            MarketRegime.SIDEWAYS: {
                'primary': 'range_trading',
                'secondary': 'mean_reversion',
                'risk_level': 'medium',
                'position_sizing': 'moderate',
                'stop_loss_type': 'support_resistance'
            },
            MarketRegime.VOLATILE: {
                'primary': 'volatility_trading',
                'secondary': 'scalping',
                'risk_level': 'low',
                'position_sizing': 'conservative',
                'stop_loss_type': 'tight_atr'
            }
        }
        
        return strategies.get(regime, strategies[MarketRegime.SIDEWAYS])
    
    def train_model(self, historical_data: pd.DataFrame, labels: List[int]):
        """Treina o modelo de ML com dados históricos"""
        try:
            # Extrai features de todo o histórico
            all_features = []
            for i in range(self.lookback_period, len(historical_data)):
                data_slice = historical_data.iloc[i-self.lookback_period:i]
                features = self._extract_regime_features(data_slice)
                feature_vector = self._prepare_features_for_ml(features)
                all_features.append(feature_vector)
            
            if len(all_features) != len(labels):
                raise ValueError("Mismatch between features and labels length")
            
            # Treina o modelo
            X = np.array(all_features)
            y = np.array(labels)
            
            # Normaliza features
            X_scaled = self.scaler.fit_transform(X)
            
            # Treina Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_scaled, y)
            
            # Salva modelo
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/regime_classifier.joblib')
            joblib.dump(self.scaler, 'models/regime_scaler.joblib')
            
            self.logger.info("Market regime model trained and saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise