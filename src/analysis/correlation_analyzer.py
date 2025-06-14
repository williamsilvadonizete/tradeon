import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

class CorrelationAnalyzer:
    """
    Analisador de correlação entre pares de trading.
    """
    
    def __init__(self, lookback_period: int = 100):
        self.logger = logging.getLogger(__name__)
        self.lookback_period = lookback_period
        self.correlation_matrix = pd.DataFrame()
        self.last_update = None
        self.correlation_threshold = 0.7  # Correlação forte acima de 0.7
        
    def update_correlations(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Atualiza a matriz de correlação com os dados mais recentes.
        """
        try:
            # Prepara dados para análise
            close_prices = {}
            for symbol, data in market_data.items():
                if 'close' in data.columns:
                    close_prices[symbol] = data['close'].tail(self.lookback_period)
            
            # Cria DataFrame com preços de fechamento
            price_df = pd.DataFrame(close_prices)
            
            # Calcula matriz de correlação
            self.correlation_matrix = price_df.corr()
            self.last_update = datetime.now()
            
            self.logger.info("Matriz de correlação atualizada com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar correlações: {str(e)}")
    
    def get_correlated_pairs(self, symbol: str) -> List[Tuple[str, float]]:
        """
        Retorna pares correlacionados com o símbolo dado.
        """
        if symbol not in self.correlation_matrix.columns:
            return []
        
        correlations = self.correlation_matrix[symbol].abs()
        correlated = correlations[correlations > self.correlation_threshold]
        correlated = correlated[correlated.index != symbol]  # Remove auto-correlação
        
        return [(pair, corr) for pair, corr in correlated.items()]
    
    def should_trade(self, symbol: str, active_trades: List[str]) -> Tuple[bool, str]:
        """
        Verifica se é seguro executar um trade considerando correlações.
        """
        if not self.correlation_matrix.empty:
            correlated_pairs = self.get_correlated_pairs(symbol)
            
            # Verifica se há trades ativos em pares correlacionados
            for pair, corr in correlated_pairs:
                if pair in active_trades:
                    return False, f"Trade bloqueado: {symbol} correlacionado com {pair} (corr: {corr:.2f})"
        
        return True, "Trade permitido: sem correlações significativas"
    
    def get_correlation_insights(self, symbol: str) -> Dict[str, Any]:
        """
        Retorna insights sobre correlações para um símbolo.
        """
        insights = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'correlated_pairs': [],
            'correlation_strength': 'low',
            'trading_risk': 'low'
        }
        
        if symbol in self.correlation_matrix.columns:
            correlations = self.correlation_matrix[symbol].abs()
            correlated = correlations[correlations > self.correlation_threshold]
            correlated = correlated[correlated.index != symbol]
            
            insights['correlated_pairs'] = [
                {'pair': pair, 'correlation': corr}
                for pair, corr in correlated.items()
            ]
            
            # Classifica força da correlação
            if len(correlated) > 3:
                insights['correlation_strength'] = 'high'
                insights['trading_risk'] = 'high'
            elif len(correlated) > 0:
                insights['correlation_strength'] = 'medium'
                insights['trading_risk'] = 'medium'
        
        return insights 