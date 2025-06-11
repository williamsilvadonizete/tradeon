import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class TradingMemory:
    def __init__(self, memory_window=10):
        self.memory_window = memory_window
        self.trades = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'average_win': 0,
            'average_loss': 0,
            'profit_factor': 0
        }
        
    def add_trade(self, trade_data):
        """
        Adiciona uma nova operação à memória.
        
        Args:
            trade_data (dict): Dados da operação
        """
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade_data)
        
        # Mantém apenas as últimas N operações
        if len(self.trades) > self.memory_window:
            self.trades.pop(0)
            
        # Atualiza métricas
        self._update_metrics()
        
    def _update_metrics(self):
        """
        Atualiza as métricas de performance.
        """
        if not self.trades:
            return
            
        # Calcula métricas básicas
        self.performance_metrics['total_trades'] = len(self.trades)
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0]
        
        self.performance_metrics['winning_trades'] = len(winning_trades)
        self.performance_metrics['losing_trades'] = len(losing_trades)
        
        # Calcula métricas de performance
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        self.performance_metrics['total_pnl'] = total_pnl
        
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            )
            
        if winning_trades:
            self.performance_metrics['average_win'] = np.mean([t.get('pnl', 0) for t in winning_trades])
            
        if losing_trades:
            self.performance_metrics['average_loss'] = np.mean([t.get('pnl', 0) for t in losing_trades])
            
        # Calcula profit factor
        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        if total_losses > 0:
            self.performance_metrics['profit_factor'] = total_wins / total_losses
        else:
            self.performance_metrics['profit_factor'] = float('inf')
            
    def get_recent_trades(self, n=None):
        """
        Retorna as N operações mais recentes.
        
        Args:
            n (int, optional): Número de operações para retornar
            
        Returns:
            list: Lista de operações recentes
        """
        if n is None:
            n = self.memory_window
            
        return self.trades[-n:]
        
    def get_performance_metrics(self):
        """
        Retorna as métricas de performance atuais.
        
        Returns:
            dict: Métricas de performance
        """
        return self.performance_metrics
        
    def save_to_file(self, filename='trading_memory.json'):
        """
        Salva a memória em um arquivo JSON.
        
        Args:
            filename (str): Nome do arquivo
        """
        data = {
            'trades': self.trades,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_from_file(self, filename='trading_memory.json'):
        """
        Carrega a memória de um arquivo JSON.
        
        Args:
            filename (str): Nome do arquivo
        """
        if not os.path.exists(filename):
            return
            
        with open(filename, 'r') as f:
            data = json.load(f)
            
        self.trades = data.get('trades', [])
        self.performance_metrics = data.get('performance_metrics', self.performance_metrics)
        
    def get_learning_insights(self):
        """
        Gera insights para aprendizado baseado no histórico de operações.
        
        Returns:
            dict: Insights de aprendizado
        """
        if not self.trades:
            return {}
            
        insights = {
            'best_performing_symbol': None,
            'best_timeframe': None,
            'best_indicator_combination': None,
            'risk_adjustment': 1.0
        }
        
        # Agrupa operações por símbolo
        symbol_performance = {}
        for trade in self.trades:
            symbol = trade.get('symbol')
            if symbol not in symbol_performance:
                symbol_performance[symbol] = []
            symbol_performance[symbol].append(trade.get('pnl', 0))
            
        # Encontra o melhor símbolo
        if symbol_performance:
            best_symbol = max(
                symbol_performance.items(),
                key=lambda x: np.mean(x[1])
            )[0]
            insights['best_performing_symbol'] = best_symbol
            
        # Ajusta o risco baseado no win rate
        win_rate = self.performance_metrics['win_rate']
        if win_rate < 0.4:
            insights['risk_adjustment'] = 0.5  # Reduz o risco
        elif win_rate > 0.6:
            insights['risk_adjustment'] = 1.5  # Aumenta o risco
            
        return insights 