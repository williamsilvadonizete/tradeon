import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import ta

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        self.rsi_period = 14
        self.sma_periods = [20, 50, 200]  # Short, medium, long term
        self.ema_periods = [12, 26]  # For MACD
        self.macd_signal_period = 9
        self.bb_period = 20
        self.bb_std = 2
        self.atr_period = 14
        self.volume_ma_period = 20
        
    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            rsi = ta.momentum.RSIIndicator(close=pd.Series(closes), window=period)
            return rsi.rsi().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0
        
    def calculate_sma(self, closes: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        try:
            sma = ta.trend.SMAIndicator(close=pd.Series(closes), window=period)
            return sma.sma_indicator().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return closes[-1]
        
    def calculate_ema(self, closes: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        try:
            ema = ta.trend.EMAIndicator(close=pd.Series(closes), window=period)
            return ema.ema_indicator().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return closes[-1]
        
    def calculate_macd(self, closes: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        try:
            macd = ta.trend.MACD(
                close=pd.Series(closes),
                window_fast=fast_period,
                window_slow=slow_period,
                window_sign=signal_period
            )
            return macd.macd().iloc[-1], macd.macd_signal().iloc[-1], macd.macd_diff().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return 0.0, 0.0, 0.0
        
    def calculate_bollinger_bands(self, closes: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        try:
            bb = ta.volatility.BollingerBands(
                close=pd.Series(closes),
                window=period,
                window_dev=std_dev
            )
            return bb.bollinger_hband().iloc[-1], bb.bollinger_mavg().iloc[-1], bb.bollinger_lband().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return closes[-1], closes[-1], closes[-1]
        
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            atr = ta.volatility.AverageTrueRange(
                high=pd.Series(highs),
                low=pd.Series(lows),
                close=pd.Series(closes),
                window=period
            )
            return atr.average_true_range().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0
        
    def calculate_volume_indicators(self, volumes: List[float], closes: List[float]) -> Tuple[float, float]:
        """Calculate volume indicators."""
        try:
            volume_ma = ta.volume.VolumeWeightedAveragePrice(
                high=pd.Series(closes),
                low=pd.Series(closes),
                close=pd.Series(closes),
                volume=pd.Series(volumes),
                window=20
            ).volume_weighted_average_price().iloc[-1]
            momentum = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0.0
            return volume_ma, momentum
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return 0.0, 0.0
        
    def analyze_trade(self, candles: List[Dict]) -> Dict:
        """
        Analyze trade based on all technical indicators
        Returns a dictionary with indicators and trade evaluation
        """
        if not candles:
            return {
                'status': 'error',
                'message': 'No candle data provided'
            }
            
        # Extract price data
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        volumes = [c['volume'] for c in candles]
        
        # Calculate indicators
        rsi = self.calculate_rsi(closes)
        sma20 = self.calculate_sma(closes, 20)
        sma50 = self.calculate_sma(closes, 50)
        sma200 = self.calculate_sma(closes, 200)
        macd_line, signal_line, histogram = self.calculate_macd(closes)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)
        atr = self.calculate_atr(highs, lows, closes)
        volume_ma, momentum = self.calculate_volume_indicators(volumes, closes)
        
        # Current price
        current_price = closes[-1]
        
        # Evaluate trade
        score = 0
        reasons = []
        
        # RSI Analysis
        if not np.isnan(rsi):
            if rsi > 70:
                score -= 2
                reasons.append("RSI indicates overbought conditions")
            elif rsi < 30:
                score += 2
                reasons.append("RSI indicates oversold conditions")
            
        # Moving Averages Analysis
        if not any(np.isnan([sma20, sma50, sma200])):
            if current_price > sma20 > sma50 > sma200:
                score += 2
                reasons.append("Strong uptrend (price above all SMAs)")
            elif current_price < sma20 < sma50 < sma200:
                score -= 2
                reasons.append("Strong downtrend (price below all SMAs)")
            
        # MACD Analysis
        if not any(np.isnan([macd_line, signal_line, histogram])):
            if macd_line > signal_line and histogram > 0:
                score += 1
                reasons.append("MACD indicates bullish momentum")
            elif macd_line < signal_line and histogram < 0:
                score -= 1
                reasons.append("MACD indicates bearish momentum")
            
        # Bollinger Bands Analysis
        if not any(np.isnan([bb_upper, bb_lower])):
            if current_price > bb_upper:
                score -= 1
                reasons.append("Price above upper Bollinger Band")
            elif current_price < bb_lower:
                score += 1
                reasons.append("Price below lower Bollinger Band")
            
        # ATR Analysis
        if not np.isnan(atr):
            avg_range = np.mean([c['high'] - c['low'] for c in candles[-5:]])
            if atr > avg_range:
                reasons.append("High volatility detected")
            
        # Volume Analysis
        if not np.isnan(volume_ma):
            if volumes[-1] > volume_ma:
                score += 1
                reasons.append("Above average volume")
            else:
                score -= 1
                reasons.append("Below average volume")
            
        # Momentum Analysis
        if not np.isnan(momentum):
            if momentum > 0:
                score += 1
                reasons.append("Positive price momentum")
            else:
                score -= 1
                reasons.append("Negative price momentum")
            
        # Final evaluation
        if score >= 3:
            evaluation = 'good'
        elif score <= -3:
            evaluation = 'bad'
        else:
            evaluation = 'neutral'
            
        # Create indicators dictionary with safe values
        indicators = {
            'rsi': float(rsi) if not np.isnan(rsi) else 50.0,
            'sma20': float(sma20) if not np.isnan(sma20) else current_price,
            'sma50': float(sma50) if not np.isnan(sma50) else current_price,
            'sma200': float(sma200) if not np.isnan(sma200) else current_price,
            'macd_line': float(macd_line) if not np.isnan(macd_line) else 0.0,
            'macd_signal': float(signal_line) if not np.isnan(signal_line) else 0.0,
            'macd_histogram': float(histogram) if not np.isnan(histogram) else 0.0,
            'bb_upper': float(bb_upper) if not np.isnan(bb_upper) else current_price * 1.1,
            'bb_middle': float(bb_middle) if not np.isnan(bb_middle) else current_price,
            'bb_lower': float(bb_lower) if not np.isnan(bb_lower) else current_price * 0.9,
            'atr': float(atr) if not np.isnan(atr) else 0.0,
            'volume_ma': float(volume_ma) if not np.isnan(volume_ma) else volumes[-1],
            'momentum': float(momentum) if not np.isnan(momentum) else 0.0,
            'current_price': float(current_price)
        }
            
        return {
            'status': 'success',
            'indicators': indicators,
            'evaluation': evaluation,
            'score': score,
            'reasons': reasons
        }
        
    def generate_signal(self, indicators: Dict) -> Tuple[str, float]:
        """
        Generate trading signal based on indicators
        Returns (action, confidence)
        """
        try:
            action = "hold"
            confidence = 0.0
            
            # RSI Analysis
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                action = "buy"
                confidence += 0.3
            elif rsi > 70:
                action = "sell"
                confidence += 0.3
                
            # Moving Averages Analysis
            current_price = indicators.get('current_price', 0)
            sma20 = indicators.get('sma20', 0)
            sma50 = indicators.get('sma50', 0)
            sma200 = indicators.get('sma200', 0)
            
            if current_price > sma20 > sma50 > sma200:
                if action == "buy":
                    confidence += 0.3
                else:
                    action = "buy"
                    confidence = 0.3
            elif current_price < sma20 < sma50 < sma200:
                if action == "sell":
                    confidence += 0.3
                else:
                    action = "sell"
                    confidence = 0.3
                    
            # MACD Analysis
            macd_line = indicators.get('macd_line', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_hist = indicators.get('macd_histogram', 0)
            
            if macd_line > macd_signal and macd_hist > 0:
                if action == "buy":
                    confidence += 0.2
                elif action == "hold":
                    action = "buy"
                    confidence = 0.2
            elif macd_line < macd_signal and macd_hist < 0:
                if action == "sell":
                    confidence += 0.2
                elif action == "hold":
                    action = "sell"
                    confidence = 0.2
                    
            # Bollinger Bands Analysis
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            if current_price < bb_lower:
                if action == "buy":
                    confidence += 0.2
                elif action == "hold":
                    action = "buy"
                    confidence = 0.2
            elif current_price > bb_upper:
                if action == "sell":
                    confidence += 0.2
                elif action == "hold":
                    action = "sell"
                    confidence = 0.2
                    
            return action, min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return "hold", 0.0 