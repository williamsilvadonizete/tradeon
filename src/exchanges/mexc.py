import ccxt
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from .base import ExchangeAdapter

class MEXCAdapter(ExchangeAdapter):
    def initialize(self) -> None:
        """
        Inicializa a conexão com a MEXC.
        """
        try:
            self.exchange = ccxt.mexc({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'sandbox': self.testnet,  # MEXC usa 'sandbox' ao invés de 'testnet'
                'options': {
                    'defaultType': 'spot',  # MEXC principalmente spot trading
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,
                    'createMarketBuyOrderRequiresPrice': False
                }
            })
            
            # Carrega os mercados
            self.exchange.load_markets()
            self.logger.info("MEXC connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing MEXC connection: {str(e)}")
            raise
            
    def get_historical_data(self, symbol: str, timeframe: str, 
                          limit: int = 100) -> pd.DataFrame:
        """
        Obtém dados históricos da MEXC.
        
        Args:
            symbol: Par de trading (ex: 'BTC/USDT')
            timeframe: Timeframe (ex: '1h', '4h', '1d')
            limit: Número de candles a retornar
            
        Returns:
            DataFrame com dados OHLCV
        """
        try:
            # Formata o símbolo para a MEXC
            if '/' not in symbol:
                # Converte BTCUSDT para BTC/USDT
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    symbol = f"{base}/{quote}"
                else:
                    self.logger.warning(f"Cannot parse symbol: {symbol}")
                    return pd.DataFrame()
            
            # Busca dados históricos
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Converte para DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_balance(self) -> Dict[str, float]:
        """
        Obtém o saldo da conta.
        
        Returns:
            Dict com saldos por moeda
        """
        try:
            balance = self.exchange.fetch_balance()
            return {currency: info['free'] for currency, info in balance.items() 
                   if isinstance(info, dict) and info.get('free', 0) > 0}
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            return {}
    
    def place_order(self, symbol: str, order_type: str, side: str, 
                   amount: float, price: Optional[float] = None) -> Dict:
        """
        Coloca uma ordem na MEXC.
        
        Args:
            symbol: Par de trading
            order_type: Tipo da ordem ('market' ou 'limit')
            side: Lado da ordem ('buy' ou 'sell')
            amount: Quantidade
            price: Preço (obrigatório para ordens limit)
            
        Returns:
            Dict com informações da ordem
        """
        try:
            # Formata o símbolo
            if '/' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    symbol = f"{base}/{quote}"
            
            # Coloca a ordem
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            else:  # limit
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            
            self.logger.info(f"Order placed: {order['id']} - {side} {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Obtém ticker de um símbolo.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Dict com informações do ticker
        """
        try:
            # Formata o símbolo
            if '/' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    symbol = f"{base}/{quote}"
            
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            return {}
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancela uma ordem.
        
        Args:
            order_id: ID da ordem
            symbol: Par de trading
            
        Returns:
            Dict com informações da ordem cancelada
        """
        try:
            # Formata o símbolo
            if '/' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    symbol = f"{base}/{quote}"
            
            result = self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order cancelled: {order_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """
        Obtém status de uma ordem.
        
        Args:
            order_id: ID da ordem
            symbol: Par de trading
            
        Returns:
            Dict com informações da ordem
        """
        try:
            # Formata o símbolo
            if '/' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    symbol = f"{base}/{quote}"
            
            order = self.exchange.fetch_order(order_id, symbol)
            return order
            
        except Exception as e:
            self.logger.error(f"Error fetching order status {order_id}: {str(e)}")
            return {}