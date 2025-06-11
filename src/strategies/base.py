from abc import ABC, abstractmethod
import pandas as pd

class TradingStrategy(ABC):
    def __init__(self, exchange_adapter):
        self.exchange = exchange_adapter
        self.indicators = {}
        
    @abstractmethod
    def calculate_indicators(self, df):
        """
        Calcula os indicadores técnicos para a estratégia.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preço
            
        Returns:
            pd.DataFrame: DataFrame com indicadores adicionados
        """
        pass
        
    @abstractmethod
    def generate_signal(self, df):
        """
        Gera um sinal de trading baseado nos indicadores.
        
        Args:
            df (pd.DataFrame): DataFrame com indicadores
            
        Returns:
            dict: Sinal de trading com:
                - action: 'buy', 'sell', ou 'hold'
                - confidence: float (0 a 1)
                - reason: str
        """
        pass
        
    @abstractmethod
    def get_position_size(self, signal, current_price, available_balance):
        """
        Calcula o tamanho da posição baseado no sinal e capital disponível.
        
        Args:
            signal (dict): Sinal de trading
            current_price (float): Preço atual
            available_balance (float): Saldo disponível
            
        Returns:
            float: Tamanho da posição
        """
        pass
        
    def execute_trade(self, symbol, signal, position_size):
        """
        Executa uma operação na exchange.
        
        Args:
            symbol (str): Par de trading
            signal (dict): Sinal de trading
            position_size (float): Tamanho da posição
            
        Returns:
            dict: Resultado da operação
        """
        try:
            if signal['action'] == 'buy':
                order = self.exchange.create_order(
                    symbol,
                    'market',
                    'buy',
                    position_size
                )
            elif signal['action'] == 'sell':
                order = self.exchange.create_order(
                    symbol,
                    'market',
                    'sell',
                    position_size
                )
            else:
                return None
                
            return {
                'symbol': symbol,
                'action': signal['action'],
                'amount': position_size,
                'order_id': order['id'],
                'status': order['status']
            }
            
        except Exception as e:
            print(f"Erro ao executar operação: {e}")
            return None 