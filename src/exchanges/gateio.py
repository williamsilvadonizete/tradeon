import ccxt
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from .base import ExchangeAdapter

class GateIOAdapter(ExchangeAdapter):
    def initialize(self) -> None:
        """
        Inicializa a conexão com a Gate.io.
        """
        try:
            self.exchange = ccxt.gateio({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'sandbox': self.testnet,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,
                    'createMarketBuyOrderRequiresPrice': False,
                    'defaultMarginMode': 'isolated'
                }
            })
            
            self.exchange.load_markets()
            self.logger.info("Gate.io connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Gate.io connection: {str(e)}")
            raise
            
    def get_historical_data(self, symbol: str, timeframe: str, 
                          limit: int = 100) -> pd.DataFrame:
        """
        Obtém dados históricos da Gate.io.
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            ohlcv = self.exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise

    def get_current_price(self, symbol: str) -> float:
        """
        Obtém o preço atual de um par de trading na Gate.io.
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            ticker = self.exchange.fetch_ticker(formatted_symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            raise

    def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None,
                    params: Dict = None) -> Dict:
        """
        Cria uma ordem na Gate.io.
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            params = params or {}
            
            if order_type.lower() == 'market':
                order = self.exchange.create_market_order(
                    formatted_symbol, side, amount, None, None, params
                )
            else:  # limit order
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = self.exchange.create_limit_order(
                    formatted_symbol, side, amount, price, None, params
                )
            
            self.logger.info(f"Order created: {order['id']} - {side} {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order: {str(e)}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancela uma ordem na Gate.io.
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            result = self.exchange.cancel_order(order_id, formatted_symbol)
            self.logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise

    def get_balance(self, currency: str) -> Dict:
        """
        Obtém o saldo de uma moeda na Gate.io.
        """
        try:
            balance = self.exchange.fetch_balance()
            if currency in balance:
                return {
                    'free': balance[currency]['free'],
                    'used': balance[currency]['used'],
                    'total': balance[currency]['total']
                }
            else:
                return {'free': 0.0, 'used': 0.0, 'total': 0.0}
        except Exception as e:
            self.logger.error(f"Error fetching balance for {currency}: {str(e)}")
            raise

    def format_symbol(self, symbol: str) -> str:
        """
        Formata o símbolo para a Gate.io.
        
        Para spot trading: BTC/USDT
        Para futures: BTC/USDT:USDT
        """
        if '/' not in symbol:
            # Converte BTCUSDT para BTC/USDT
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                quote = 'USDT'
                symbol = f"{base}/{quote}"
        
        return symbol
    
    def set_futures_mode(self):
        """
        Configura o exchange para modo futures.
        """
        try:
            self.exchange.options['defaultType'] = 'future'
            self.logger.info("Switched to futures trading mode")
        except Exception as e:
            self.logger.error(f"Error switching to futures mode: {str(e)}")
            raise
    
    def set_spot_mode(self):
        """
        Configura o exchange para modo spot.
        """
        try:
            self.exchange.options['defaultType'] = 'spot'
            self.logger.info("Switched to spot trading mode")
        except Exception as e:
            self.logger.error(f"Error switching to spot mode: {str(e)}")
            raise

    def place_futures_order(self, symbol: str, order_type: str, side: str, 
                           amount: float, price: Optional[float] = None, 
                           leverage: int = 1) -> Dict:
        """
        Coloca uma ordem de futuros na Gate.io.
        """
        try:
            # Muda para modo futures
            self.set_futures_mode()
            
            # Formata símbolo para futuros (adiciona :USDT)
            if '/' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    symbol = f"{base}/{quote}:USDT"
            elif ':' not in symbol:
                symbol = f"{symbol}:USDT"
            
            # Parâmetros para futuros
            params = {
                'leverage': leverage,
                'marginMode': 'isolated'  # Margem isolada sempre
            }
            
            if order_type.lower() == 'market':
                order = self.exchange.create_market_order(symbol, side, amount, None, None, params)
            else:  # limit order
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = self.exchange.create_limit_order(symbol, side, amount, price, None, params)
            
            self.logger.info(f"Futures order placed: {order['id']} - {side} {amount} {symbol} @ {leverage}x leverage")
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing futures order: {str(e)}")
            raise

    def get_futures_balance(self) -> Dict:
        """
        Obtém saldo da conta de futuros.
        """
        try:
            self.set_futures_mode()
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching futures balance: {str(e)}")
            raise

    def set_leverage(self, symbol: str, leverage: int, margin_mode: str = 'isolated'):
        """
        Define alavancagem para um símbolo.
        """
        try:
            self.set_futures_mode()
            
            # Formata símbolo para futuros
            if ':' not in symbol:
                symbol = f"{symbol}:USDT"
            
            result = self.exchange.set_leverage(leverage, symbol, {
                'marginMode': margin_mode
            })
            
            self.logger.info(f"Leverage set to {leverage}x for {symbol} (margin: {margin_mode})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            raise