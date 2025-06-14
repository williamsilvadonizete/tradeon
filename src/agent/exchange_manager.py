import os
import logging
import asyncio
from typing import Dict, List, Optional
import ccxt.async_support as ccxt
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExchangeManager:
    """Manages exchange connections and operations"""
    
    def __init__(self):
        """Initialize the exchange manager."""
        self.logger = logging.getLogger(__name__)
        self.exchange_id = os.getenv('EXCHANGE_ID', 'gateio').lower()
        self.exchange = None
        self.trading_pairs = os.getenv('TRADING_PAIRS', 'BTCUSDT,ETHUSDT,SOLUSDT,AAVEUSDT,SUIUSDT,BNBUSDT').split(',')
        self.logger.info(f"Initialized trading pairs: {self.trading_pairs}")
        
    async def initialize(self):
        """Initialize exchange connection."""
        try:
            # Get exchange configuration
            api_key = os.getenv('EXCHANGE_API_KEY')
            api_secret = os.getenv('EXCHANGE_SECRET')
            testnet = os.getenv('EXCHANGE_TESTNET', 'false').lower() == 'true'
            
            if not api_key or not api_secret:
                self.logger.warning("Exchange API credentials not found in environment variables")
                return
            
            # Initialize exchange
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'testnet': testnet
                }
            })
            
            # Load markets to ensure connection is working
            await self.exchange.load_markets()
            
            self.logger.info(f"Exchange {self.exchange_id} initialized successfully")
            self.logger.info(f"Testnet mode: {'Enabled' if testnet else 'Disabled'}")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {str(e)}")
            raise

    def is_initialized(self) -> bool:
        """Check if exchange is initialized."""
        return self.exchange is not None

    def get_trading_pairs(self) -> List[str]:
        """Get list of available trading pairs."""
        return self.trading_pairs  # Return the pairs from environment variable

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe format to Gate.io format."""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }
        return timeframe_map.get(timeframe, '1h')  # Default to 1h if invalid

    async def fetch_historical_data(self, symbol: str, timeframe: str = '2h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data for a symbol."""
        try:
            if not self.exchange:
                self.logger.error("Exchange not initialized")
                return None

            # Verify if symbol ends with USDT
            if not symbol.endswith('USDT'):
                self.logger.error(f"Invalid symbol format: {symbol}. Must end with USDT")
                return None

            # Convert symbol format (e.g., BTCUSDT -> BTC/USDT)
            formatted_symbol = symbol.replace('USDT', '/USDT')
            
            # Convert timeframe to Gate.io format
            gate_timeframe = self._convert_timeframe(timeframe)
            
            self.logger.info(f"Fetching historical data for {formatted_symbol} on {self.exchange_id} with timeframe {gate_timeframe}")
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=formatted_symbol,
                timeframe=gate_timeframe,
                limit=limit
            )
            
            if not ohlcv:
                self.logger.warning(f"No data returned for {formatted_symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return None

    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None

    async def get_ticker(self, symbol: str, exchange_id: str) -> Optional[Dict]:
        """Get ticker data for a symbol from a specific exchange."""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                self.logger.warning(f"Exchange {exchange_id} not found")
                return None
            
            ticker = await exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol} on {exchange_id}: {str(e)}")
            return None
    
    async def create_order(self, symbol: str, exchange_id: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict]:
        """Create an order on a specific exchange."""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                self.logger.warning(f"Exchange {exchange_id} not found")
                return None
            
            if order_type == 'market':
                order = await exchange.create_market_order(symbol, side, amount)
            else:  # limit
                if price is None:
                    self.logger.error("Price is required for limit orders")
                    return None
                order = await exchange.create_limit_order(symbol, side, amount, price)
                
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order on {exchange_id}: {str(e)}")
            return None
    
    async def get_balance(self, exchange_id: str) -> Optional[float]:
        """Get USDT balance from a specific exchange."""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                self.logger.warning(f"Exchange {exchange_id} not found")
                return None
            
            balance = await exchange.fetch_balance()
            if not balance or 'USDT' not in balance:
                self.logger.warning(f"No USDT balance found for {exchange_id}")
                return None
                
            return float(balance['USDT']['free'])
            
        except Exception as e:
            self.logger.error(f"Error fetching balance from {exchange_id}: {str(e)}")
            return None
    
    def get_active_exchanges(self) -> List[str]:
        """Get list of active exchange IDs."""
        return [self.exchange_id] if self.exchange else []

    def is_exchange_active(self, exchange_id: str) -> bool:
        """Check if an exchange is active."""
        return exchange_id in self.exchanges 
    
    async def get_historical_data(self, symbol: str, exchange_id: str, timeframe: str = '1h', limit: int = 100) -> List[Dict]:
        """
        Get historical candle data for technical analysis.
        
        Args:
            symbol: Trading pair symbol
            exchange_id: Exchange identifier
            timeframe: Candle timeframe (default: 1h)
            limit: Number of candles to fetch (default: 100)
            
        Returns:
            List of candles with OHLCV data
        """
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                self.logger.error(f"Exchange {exchange_id} not initialized")
                return None
                
            # Fetch OHLCV data
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to list of dictionaries
            candles = []
            for candle in ohlcv:
                candles.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
                
            self.logger.info(f"Fetched {len(candles)} candles for {symbol} on {exchange_id}")
            return candles
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return None 