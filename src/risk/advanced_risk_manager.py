import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json

from ..analysis.market_regime import MarketRegime, MarketRegimeDetector

class RiskLevel(Enum):
    """Níveis de risco do portfólio"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class PositionRisk:
    """Informações de risco de uma posição"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    risk_amount: float
    risk_percentage: float
    correlation_exposure: float
    volatility_adjustment: float

@dataclass
class PortfolioRiskMetrics:
    """Métricas de risco do portfólio"""
    total_exposure: float
    total_risk: float
    correlation_risk: float
    concentration_risk: float
    volatility_risk: float
    drawdown_risk: float
    risk_level: RiskLevel
    max_additional_risk: float

class AdvancedRiskManager:
    """
    Sistema avançado de gerenciamento de risco com:
    - Kelly Criterion otimizado
    - Gestão de correlação
    - Control de portfolio heat
    - Position sizing dinâmico
    - Proteção contra drawdown
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.regime_detector = MarketRegimeDetector()
        
        # Configurações padrão
        self.config = {
            'max_portfolio_risk': 0.20,  # 20% máximo de risco do portfólio
            'max_single_position_risk': 0.05,  # 5% máximo por posição
            'max_correlated_exposure': 0.15,  # 15% máximo em ativos correlacionados
            'max_sector_exposure': 0.25,  # 25% máximo por setor
            'kelly_safety_factor': 0.25,  # Usar apenas 25% do Kelly Criterion
            'volatility_adjustment_factor': 0.5,  # Fator de ajuste por volatilidade
            'drawdown_protection_threshold': 0.10,  # 10% drawdown para proteção
            'correlation_threshold': 0.7,  # Correlação de 70% para considerar alto risco
            'rebalance_threshold': 0.05,  # 5% mudança para rebalanceamento
            'min_position_size': 0.001,  # Tamanho mínimo de posição
            'max_positions': 10  # Máximo de posições simultâneas
        }
        
        if config:
            self.config.update(config)
        
        # Estado do portfólio
        self.portfolio_value = 10000.0  # Valor inicial
        self.current_positions = {}
        self.correlation_matrix = {}
        self.volatility_cache = {}
        self.performance_history = []
        
        # Métricas de performance
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_equity': self.portfolio_value,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def calculate_optimal_position_size(self, 
                                      signal: Dict,
                                      current_price: float,
                                      market_regime: MarketRegime,
                                      market_data: pd.DataFrame) -> Dict:
        """
        Calcula o tamanho ótimo da posição usando múltiplos fatores.
        
        Args:
            signal: Sinal de trading com action e confidence
            current_price: Preço atual do ativo
            market_regime: Regime atual de mercado
            market_data: Dados históricos para análise
            
        Returns:
            Dict com informações de sizing e risco
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            confidence = signal.get('confidence', 0.5)
            
            # 1. Base Kelly Criterion
            kelly_size = self._calculate_kelly_position_size(symbol, confidence)
            
            # 2. Ajuste por regime de mercado
            regime_adjustment = self._get_regime_adjustment(market_regime)
            
            # 3. Ajuste por volatilidade
            volatility_adjustment = self._calculate_volatility_adjustment(market_data)
            
            # 4. Ajuste por correlação
            correlation_adjustment = self._calculate_correlation_adjustment(symbol)
            
            # 5. Ajuste por concentração
            concentration_adjustment = self._calculate_concentration_adjustment()
            
            # 6. Ajuste por drawdown atual
            drawdown_adjustment = self._calculate_drawdown_adjustment()
            
            # 7. Ajuste por liquidez
            liquidity_adjustment = self._calculate_liquidity_adjustment(symbol, market_data)
            
            # Combina todos os ajustes
            base_size = self.portfolio_value * self.config['max_single_position_risk']
            
            final_size = (kelly_size * 
                         regime_adjustment * 
                         volatility_adjustment * 
                         correlation_adjustment * 
                         concentration_adjustment * 
                         drawdown_adjustment * 
                         liquidity_adjustment)
            
            # Aplica limites
            final_size = max(final_size, self.config['min_position_size'] * self.portfolio_value)
            final_size = min(final_size, base_size)
            
            # Converte para quantidade
            quantity = final_size / current_price
            
            # Calcula métricas de risco
            risk_metrics = self._calculate_position_risk_metrics(
                symbol, quantity, current_price, signal
            )
            
            return {
                'quantity': quantity,
                'position_value': final_size,
                'risk_amount': risk_metrics['risk_amount'],
                'risk_percentage': risk_metrics['risk_percentage'],
                'adjustments': {
                    'kelly_base': kelly_size,
                    'regime': regime_adjustment,
                    'volatility': volatility_adjustment,
                    'correlation': correlation_adjustment,
                    'concentration': concentration_adjustment,
                    'drawdown': drawdown_adjustment,
                    'liquidity': liquidity_adjustment
                },
                'risk_metrics': risk_metrics,
                'approved': risk_metrics['risk_percentage'] <= self.config['max_single_position_risk']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'approved': False,
                'error': str(e)
            }
    
    def _calculate_kelly_position_size(self, symbol: str, confidence: float) -> float:
        """Calcula position size usando Kelly Criterion otimizado"""
        try:
            # Obtém histórico de performance para o símbolo
            symbol_trades = [t for t in self.performance_history if t.get('symbol') == symbol]
            
            if len(symbol_trades) < 10:  # Poucos dados, usa métricas globais
                win_rate = self.performance_metrics['win_rate']
                avg_win = self.performance_metrics['avg_win']
                avg_loss = abs(self.performance_metrics['avg_loss'])
            else:
                # Calcula métricas específicas do símbolo
                wins = [t for t in symbol_trades if t['pnl'] > 0]
                losses = [t for t in symbol_trades if t['pnl'] < 0]
                
                win_rate = len(wins) / len(symbol_trades)
                avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.02
                avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0.02
            
            # Kelly Criterion: f = (bp - q) / b
            # onde: b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss == 0 or win_rate == 0:
                return self.config['min_position_size']
            
            b = avg_win / avg_loss  # Razão gain/loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Ajusta pela confiança do sinal
            confidence_adjusted_kelly = kelly_fraction * confidence
            
            # Aplica fator de segurança
            safe_kelly = confidence_adjusted_kelly * self.config['kelly_safety_factor']
            
            # Garante que está dentro dos limites
            safe_kelly = max(0, min(safe_kelly, self.config['max_single_position_risk']))
            
            return safe_kelly * self.portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {str(e)}")
            return self.config['min_position_size'] * self.portfolio_value
    
    def _get_regime_adjustment(self, regime: MarketRegime) -> float:
        """Ajuste baseado no regime de mercado"""
        adjustments = {
            MarketRegime.BULLISH: 1.2,      # Mais agressivo em alta
            MarketRegime.BEARISH: 0.6,      # Mais conservador em baixa
            MarketRegime.SIDEWAYS: 0.8,     # Moderado em lateral
            MarketRegime.VOLATILE: 0.4      # Muito conservador em volatilidade
        }
        return adjustments.get(regime, 0.8)
    
    def _calculate_volatility_adjustment(self, market_data: pd.DataFrame) -> float:
        """Ajuste baseado na volatilidade atual"""
        try:
            # Calcula volatilidade realizada (20 períodos)
            returns = market_data['close'].pct_change().dropna()
            current_vol = returns.tail(20).std() * np.sqrt(252)  # Anualizada
            
            # Volatilidade histórica (100 períodos)
            historical_vol = returns.tail(100).std() * np.sqrt(252)
            
            if historical_vol == 0:
                return 1.0
            
            vol_ratio = current_vol / historical_vol
            
            # Reduz posição quando volatilidade está alta
            if vol_ratio > 2.0:
                return 0.5  # Metade do tamanho
            elif vol_ratio > 1.5:
                return 0.7
            elif vol_ratio < 0.5:
                return 1.3  # Pode aumentar um pouco quando vol é baixa
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 1.0
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Ajuste baseado na correlação com posições existentes"""
        try:
            if not self.current_positions:
                return 1.0
            
            total_correlation_risk = 0
            for existing_symbol, position in self.current_positions.items():
                if existing_symbol == symbol:
                    continue
                
                # Obtém correlação entre os ativos
                correlation = self._get_correlation(symbol, existing_symbol)
                
                if abs(correlation) > self.config['correlation_threshold']:
                    # Calcula exposição correlacionada
                    corr_exposure = position['risk_amount'] * abs(correlation)
                    total_correlation_risk += corr_exposure
            
            # Se exposição correlacionada é alta, reduz posição
            correlation_ratio = total_correlation_risk / self.portfolio_value
            
            if correlation_ratio > self.config['max_correlated_exposure']:
                return 0.3  # Reduz drasticamente
            elif correlation_ratio > self.config['max_correlated_exposure'] * 0.7:
                return 0.6  # Reduz moderadamente
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating correlation adjustment: {str(e)}")
            return 1.0
    
    def _calculate_concentration_adjustment(self) -> float:
        """Ajuste baseado na concentração atual do portfólio"""
        try:
            if len(self.current_positions) >= self.config['max_positions']:
                return 0.5  # Reduz novas posições se já tem muitas
            
            # Calcula concentração por posição
            position_count = len(self.current_positions)
            
            if position_count == 0:
                return 1.0
            elif position_count < 3:
                return 1.1  # Pode ser um pouco mais agressivo
            elif position_count < 6:
                return 1.0
            else:
                return 0.8  # Reduz para evitar over-diversification
                
        except Exception as e:
            self.logger.error(f"Error calculating concentration adjustment: {str(e)}")
            return 1.0
    
    def _calculate_drawdown_adjustment(self) -> float:
        """Ajuste baseado no drawdown atual"""
        try:
            current_dd = self.performance_metrics['current_drawdown']
            threshold = self.config['drawdown_protection_threshold']
            
            if current_dd > threshold:
                # Reduz agressivamente durante drawdown alto
                reduction_factor = (current_dd - threshold) / threshold
                return max(0.2, 1.0 - reduction_factor)
            elif current_dd > threshold * 0.5:
                # Reduz moderadamente
                return 0.7
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating drawdown adjustment: {str(e)}")
            return 1.0
    
    def _calculate_liquidity_adjustment(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Ajuste baseado na liquidez do ativo"""
        try:
            # Analisa volume médio
            avg_volume = market_data['volume'].tail(20).mean()
            current_volume = market_data['volume'].iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Se volume está muito baixo, reduz posição
            if volume_ratio < 0.3:
                return 0.5
            elif volume_ratio < 0.6:
                return 0.8
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating liquidity adjustment: {str(e)}")
            return 1.0
    
    def _calculate_position_risk_metrics(self, symbol: str, quantity: float, 
                                       current_price: float, signal: Dict) -> Dict:
        """Calcula métricas de risco da posição"""
        try:
            position_value = quantity * current_price
            
            # Estimativa de stop loss (2% padrão se não especificado)
            stop_loss_pct = signal.get('stop_loss_pct', 0.02)
            risk_per_share = current_price * stop_loss_pct
            total_risk = quantity * risk_per_share
            risk_percentage = total_risk / self.portfolio_value
            
            return {
                'position_value': position_value,
                'risk_amount': total_risk,
                'risk_percentage': risk_percentage,
                'stop_loss_distance': stop_loss_pct,
                'position_size_pct': position_value / self.portfolio_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk metrics: {str(e)}")
            return {
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'stop_loss_distance': 0,
                'position_size_pct': 0
            }
    
    def evaluate_portfolio_risk(self) -> PortfolioRiskMetrics:
        """Avalia o risco total do portfólio"""
        try:
            total_exposure = sum(pos['position_value'] for pos in self.current_positions.values())
            total_risk = sum(pos['risk_amount'] for pos in self.current_positions.values())
            
            # Calcula risco de correlação
            correlation_risk = self._calculate_total_correlation_risk()
            
            # Calcula risco de concentração
            concentration_risk = self._calculate_concentration_risk()
            
            # Calcula risco de volatilidade
            volatility_risk = self._calculate_volatility_risk()
            
            # Determina nível de risco
            risk_level = self._determine_risk_level(total_risk / self.portfolio_value)
            
            # Calcula capacidade adicional de risco
            max_additional_risk = max(0, 
                self.config['max_portfolio_risk'] * self.portfolio_value - total_risk
            )
            
            return PortfolioRiskMetrics(
                total_exposure=total_exposure,
                total_risk=total_risk,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                volatility_risk=volatility_risk,
                drawdown_risk=self.performance_metrics['current_drawdown'],
                risk_level=risk_level,
                max_additional_risk=max_additional_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating portfolio risk: {str(e)}")
            return PortfolioRiskMetrics(
                total_exposure=0, total_risk=0, correlation_risk=0,
                concentration_risk=0, volatility_risk=0, drawdown_risk=0,
                risk_level=RiskLevel.CONSERVATIVE, max_additional_risk=0
            )
    
    def _calculate_total_correlation_risk(self) -> float:
        """Calcula risco total de correlação do portfólio"""
        total_corr_risk = 0
        symbols = list(self.current_positions.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = self._get_correlation(symbol1, symbol2)
                if abs(correlation) > self.config['correlation_threshold']:
                    pos1_risk = self.current_positions[symbol1]['risk_amount']
                    pos2_risk = self.current_positions[symbol2]['risk_amount']
                    corr_risk = (pos1_risk + pos2_risk) * abs(correlation) * 0.5
                    total_corr_risk += corr_risk
        
        return total_corr_risk
    
    def _calculate_concentration_risk(self) -> float:
        """Calcula risco de concentração"""
        if not self.current_positions:
            return 0
        
        position_sizes = [pos['position_value'] for pos in self.current_positions.values()]
        total_value = sum(position_sizes)
        
        if total_value == 0:
            return 0
        
        # Calcula índice Herfindahl (concentração)
        weights = [size / total_value for size in position_sizes]
        herfindahl_index = sum(w**2 for w in weights)
        
        # Normaliza para 0-1 (1 = máxima concentração)
        n_positions = len(self.current_positions)
        min_herfindahl = 1 / n_positions if n_positions > 0 else 1
        normalized_concentration = (herfindahl_index - min_herfindahl) / (1 - min_herfindahl)
        
        return normalized_concentration
    
    def _calculate_volatility_risk(self) -> float:
        """Calcula risco agregado de volatilidade"""
        if not self.current_positions:
            return 0
        
        total_vol_risk = 0
        for symbol, position in self.current_positions.items():
            # Obtém volatilidade do cache ou calcula
            volatility = self.volatility_cache.get(symbol, 0.2)  # 20% padrão
            position_vol_risk = position['position_value'] * volatility
            total_vol_risk += position_vol_risk
        
        return total_vol_risk / self.portfolio_value
    
    def _determine_risk_level(self, risk_ratio: float) -> RiskLevel:
        """Determina nível de risco baseado na exposição"""
        if risk_ratio < 0.05:
            return RiskLevel.CONSERVATIVE
        elif risk_ratio < 0.10:
            return RiskLevel.MODERATE
        elif risk_ratio < 0.15:
            return RiskLevel.AGGRESSIVE
        else:
            return RiskLevel.MAXIMUM
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Obtém correlação entre dois símbolos"""
        # Implementação simplificada - em produção, usar dados reais
        pair_key = f"{symbol1}_{symbol2}"
        reverse_key = f"{symbol2}_{symbol1}"
        
        if pair_key in self.correlation_matrix:
            return self.correlation_matrix[pair_key]
        elif reverse_key in self.correlation_matrix:
            return self.correlation_matrix[reverse_key]
        else:
            # Correlações estimadas para criptomoedas
            crypto_correlations = {
                ('BTCUSDT', 'ETHUSDT'): 0.8,
                ('BTCUSDT', 'SOLUSDT'): 0.7,
                ('ETHUSDT', 'SOLUSDT'): 0.75,
                ('BTCUSDT', 'BNBUSDT'): 0.6,
                ('ETHUSDT', 'AAVEUSDT'): 0.65
            }
            
            correlation = crypto_correlations.get((symbol1, symbol2), 
                         crypto_correlations.get((symbol2, symbol1), 0.3))
            
            self.correlation_matrix[pair_key] = correlation
            return correlation
    
    def update_position(self, symbol: str, position_data: Dict):
        """Atualiza informações de uma posição"""
        self.current_positions[symbol] = position_data
        self.logger.info(f"Updated position for {symbol}: {position_data}")
    
    def close_position(self, symbol: str, exit_price: float, pnl: float):
        """Fecha uma posição e atualiza métricas"""
        if symbol in self.current_positions:
            position = self.current_positions.pop(symbol)
            
            # Atualiza métricas de performance
            self._update_performance_metrics(symbol, position, exit_price, pnl)
            
            self.logger.info(f"Closed position {symbol} with PnL: ${pnl:.2f}")
    
    def _update_performance_metrics(self, symbol: str, position: Dict, 
                                  exit_price: float, pnl: float):
        """Atualiza métricas de performance"""
        try:
            # Adiciona trade ao histórico
            trade_record = {
                'symbol': symbol,
                'entry_price': position.get('entry_price', 0),
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl / position.get('position_value', 1),
                'timestamp': datetime.now(),
                'position_size': position.get('quantity', 0)
            }
            self.performance_history.append(trade_record)
            
            # Atualiza métricas agregadas
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['total_pnl'] += pnl
            
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
            
            # Atualiza valor do portfólio
            self.portfolio_value += pnl
            
            # Atualiza peak equity e drawdown
            if self.portfolio_value > self.performance_metrics['peak_equity']:
                self.performance_metrics['peak_equity'] = self.portfolio_value
                self.performance_metrics['current_drawdown'] = 0
            else:
                current_dd = (self.performance_metrics['peak_equity'] - self.portfolio_value) / self.performance_metrics['peak_equity']
                self.performance_metrics['current_drawdown'] = current_dd
                self.performance_metrics['max_drawdown'] = max(self.performance_metrics['max_drawdown'], current_dd)
            
            # Recalcula métricas derivadas
            self._recalculate_derived_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _recalculate_derived_metrics(self):
        """Recalcula métricas derivadas"""
        try:
            total_trades = self.performance_metrics['total_trades']
            
            if total_trades > 0:
                # Win rate
                self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / total_trades
                
                # Average win/loss
                wins = [t['pnl'] for t in self.performance_history if t['pnl'] > 0]
                losses = [t['pnl'] for t in self.performance_history if t['pnl'] < 0]
                
                self.performance_metrics['avg_win'] = np.mean(wins) if wins else 0
                self.performance_metrics['avg_loss'] = np.mean(losses) if losses else 0
                
                # Profit factor
                total_wins = sum(wins) if wins else 0
                total_losses = abs(sum(losses)) if losses else 1
                self.performance_metrics['profit_factor'] = total_wins / total_losses
                
                # Sharpe ratio (simplified)
                if len(self.performance_history) > 10:
                    returns = [t['pnl_pct'] for t in self.performance_history[-252:]]  # Last year
                    if returns:
                        mean_return = np.mean(returns)
                        std_return = np.std(returns)
                        self.performance_metrics['sharpe_ratio'] = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
                        
        except Exception as e:
            self.logger.error(f"Error recalculating derived metrics: {str(e)}")
    
    def get_risk_report(self) -> Dict:
        """Gera relatório completo de risco"""
        portfolio_risk = self.evaluate_portfolio_risk()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'portfolio_risk': {
                'total_exposure': portfolio_risk.total_exposure,
                'total_risk': portfolio_risk.total_risk,
                'risk_percentage': portfolio_risk.total_risk / self.portfolio_value,
                'correlation_risk': portfolio_risk.correlation_risk,
                'concentration_risk': portfolio_risk.concentration_risk,
                'volatility_risk': portfolio_risk.volatility_risk,
                'drawdown_risk': portfolio_risk.drawdown_risk,
                'risk_level': portfolio_risk.risk_level.value,
                'max_additional_risk': portfolio_risk.max_additional_risk
            },
            'performance_metrics': self.performance_metrics,
            'current_positions': {
                symbol: {
                    'value': pos['position_value'],
                    'risk': pos['risk_amount'],
                    'risk_pct': pos['risk_amount'] / self.portfolio_value
                }
                for symbol, pos in self.current_positions.items()
            },
            'risk_limits': {
                'max_portfolio_risk': self.config['max_portfolio_risk'],
                'max_single_position_risk': self.config['max_single_position_risk'],
                'max_correlated_exposure': self.config['max_correlated_exposure'],
                'max_positions': self.config['max_positions']
            }
        }
    
    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """Determina se deve reduzir exposição"""
        portfolio_risk = self.evaluate_portfolio_risk()
        risk_ratio = portfolio_risk.total_risk / self.portfolio_value
        
        # Verifica múltiplos fatores de risco
        reasons = []
        
        if risk_ratio > self.config['max_portfolio_risk']:
            reasons.append(f"Portfolio risk too high: {risk_ratio:.1%}")
        
        if portfolio_risk.correlation_risk > self.config['max_correlated_exposure'] * self.portfolio_value:
            reasons.append("Correlation risk exceeded")
        
        if self.performance_metrics['current_drawdown'] > self.config['drawdown_protection_threshold']:
            reasons.append(f"Drawdown protection triggered: {self.performance_metrics['current_drawdown']:.1%}")
        
        if portfolio_risk.concentration_risk > 0.8:
            reasons.append("High concentration risk")
        
        should_reduce = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "All risk metrics within limits"
        
        return should_reduce, reason_text