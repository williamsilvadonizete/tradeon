import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

class VolumeAnalyzer:
    """
    Analisador avançado de volume para trading.
    """
    
    def __init__(self, lookback_period: int = 100):
        self.logger = logging.getLogger(__name__)
        self.lookback_period = lookback_period
        self.volume_profiles = {}
        self.last_update = None
        
    def update_volume_profiles(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Atualiza os perfis de volume para todos os símbolos.
        """
        try:
            for symbol, data in market_data.items():
                if 'volume' in data.columns:
                    # Calcula métricas de volume
                    volume_data = data['volume'].tail(self.lookback_period)
                    
                    self.volume_profiles[symbol] = {
                        'avg_volume': volume_data.mean(),
                        'std_volume': volume_data.std(),
                        'max_volume': volume_data.max(),
                        'min_volume': volume_data.min(),
                        'volume_trend': self._calculate_volume_trend(volume_data),
                        'volume_momentum': self._calculate_volume_momentum(volume_data),
                        'last_update': datetime.now()
                    }
            
            self.last_update = datetime.now()
            self.logger.info("Perfis de volume atualizados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar perfis de volume: {str(e)}")
    
    def _calculate_volume_trend(self, volume_data: pd.Series) -> float:
        """
        Calcula a tendência do volume usando regressão linear.
        """
        x = np.arange(len(volume_data))
        y = volume_data.values
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def _calculate_volume_momentum(self, volume_data: pd.Series) -> float:
        """
        Calcula o momentum do volume (média móvel exponencial).
        """
        ema = volume_data.ewm(span=20, adjust=False).mean()
        return (ema.iloc[-1] - ema.iloc[-2]) / ema.iloc[-2] * 100
    
    def analyze_volume(self, symbol: str, current_volume: float) -> Dict[str, Any]:
        """
        Analisa o volume atual em relação ao perfil histórico.
        """
        if symbol not in self.volume_profiles:
            return {
                'status': 'unknown',
                'message': 'Perfil de volume não disponível',
                'metrics': {}
            }
        
        profile = self.volume_profiles[symbol]
        
        # Calcula métricas relativas
        volume_ratio = current_volume / profile['avg_volume']
        z_score = (current_volume - profile['avg_volume']) / profile['std_volume']
        
        # Determina status do volume
        if volume_ratio > 2.0 or z_score > 2.0:
            status = 'extremely_high'
        elif volume_ratio > 1.5 or z_score > 1.5:
            status = 'high'
        elif volume_ratio < 0.5 or z_score < -1.5:
            status = 'low'
        else:
            status = 'normal'
        
        # Analisa tendência e momentum
        trend_strength = 'strong' if abs(profile['volume_trend']) > profile['std_volume'] else 'weak'
        momentum_direction = 'up' if profile['volume_momentum'] > 0 else 'down'
        
        return {
            'status': status,
            'message': self._generate_volume_message(status, volume_ratio, trend_strength, momentum_direction),
            'metrics': {
                'volume_ratio': volume_ratio,
                'z_score': z_score,
                'trend_strength': trend_strength,
                'momentum_direction': momentum_direction,
                'volume_trend': profile['volume_trend'],
                'volume_momentum': profile['volume_momentum']
            }
        }
    
    def _generate_volume_message(self, status: str, ratio: float, trend: str, momentum: str) -> str:
        """
        Gera mensagem descritiva sobre o volume.
        """
        messages = {
            'extremely_high': f"Volume extremamente alto ({ratio:.1f}x média) com tendência {trend} e momentum {momentum}",
            'high': f"Volume acima da média ({ratio:.1f}x) com tendência {trend} e momentum {momentum}",
            'normal': f"Volume normal ({ratio:.1f}x média) com tendência {trend} e momentum {momentum}",
            'low': f"Volume abaixo da média ({ratio:.1f}x) com tendência {trend} e momentum {momentum}"
        }
        return messages.get(status, "Status de volume desconhecido")
    
    def get_volume_insights(self, symbol: str) -> Dict[str, Any]:
        """
        Retorna insights completos sobre o volume para um símbolo.
        """
        if symbol not in self.volume_profiles:
            return {
                'symbol': symbol,
                'status': 'unknown',
                'message': 'Perfil de volume não disponível',
                'metrics': {}
            }
        
        profile = self.volume_profiles[symbol]
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'status': 'available',
            'metrics': {
                'average_volume': profile['avg_volume'],
                'volume_volatility': profile['std_volume'],
                'volume_trend': profile['volume_trend'],
                'volume_momentum': profile['volume_momentum'],
                'max_volume': profile['max_volume'],
                'min_volume': profile['min_volume']
            },
            'analysis': {
                'trend_strength': 'strong' if abs(profile['volume_trend']) > profile['std_volume'] else 'weak',
                'momentum_direction': 'up' if profile['volume_momentum'] > 0 else 'down',
                'volume_health': 'healthy' if profile['volume_trend'] > 0 else 'concerning'
            }
        } 