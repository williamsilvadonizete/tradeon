from abc import ABC, abstractmethod
import ccxt
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

class ExchangeAdapter(ABC):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Inicializa o adaptador da exchange.
        
        Args:
            api_key: Chave da API
            api_secret: Segredo da API
            testnet: Se deve usar o ambiente de teste
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange = None
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Inicializa a conexão com a exchange.
        Deve ser implementado por cada adaptador específico.
        """
        pass
        
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, 
                          limit: int = 100) -> pd.DataFrame:
        """
        Obtém dados históricos de um par de trading.
        
        Args:
            symbol: Par de trading (ex: 'BTC/USDT')
            timeframe: Timeframe (ex: '1h', '4h', '1d')
            limit: Número de candles a retornar
            
        Returns:
            DataFrame com dados OHLCV
        """
        pass
        
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Obtém o preço atual de um par de trading.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Preço atual
        """
        pass
        
    @abstractmethod
    def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None,
                    params: Dict = None) -> Dict:
        """
        Cria uma ordem na exchange.
        
        Args:
            symbol: Par de trading
            order_type: Tipo de ordem ('limit', 'market')
            side: Lado da ordem ('buy', 'sell')
            amount: Quantidade
            price: Preço (opcional para ordens market)
            params: Parâmetros adicionais
            
        Returns:
            Dicionário com informações da ordem
        """
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancela uma ordem existente.
        
        Args:
            order_id: ID da ordem
            symbol: Par de trading
            
        Returns:
            Dicionário com informações da ordem cancelada
        """
        pass
        
    @abstractmethod
    def get_balance(self, currency: str) -> Dict:
        """
        Obtém o saldo de uma moeda.
        
        Args:
            currency: Símbolo da moeda
            
        Returns:
            Dicionário com informações do saldo
        """
        pass
        
    @abstractmethod
    def format_symbol(self, symbol: str) -> str:
        """
        Formata o símbolo do par de trading conforme exigido pela exchange.
        
        Args:
            symbol: Par de trading no formato padrão (ex: 'BTC/USDT')
            
        Returns:
            Símbolo formatado para a exchange
        """
        pass
        
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """
        Obtém o status de uma ordem.
        
        Args:
            order_id: ID da ordem
            symbol: Par de trading
            
        Returns:
            Dicionário com status da ordem
        """
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            self.logger.error(f"Error fetching order status: {str(e)}")
            raise
            
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtém todas as ordens abertas.
        
        Args:
            symbol: Par de trading (opcional)
            
        Returns:
            Lista de ordens abertas
        """
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {str(e)}")
            raise
            
    def get_trading_fees(self, symbol: str) -> Dict:
        """
        Obtém as taxas de trading para um par.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Dicionário com informações das taxas
        """
        try:
            return self.exchange.fetch_trading_fees(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching trading fees: {str(e)}")
            raise
            
    def get_market_info(self, symbol: str) -> Dict:
        """
        Obtém informações do mercado para um par.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Dicionário com informações do mercado
        """
        try:
            return self.exchange.fetch_market(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching market info: {str(e)}")
            raise 