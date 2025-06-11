import ccxt
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from .base import ExchangeAdapter

class BybitAdapter(ExchangeAdapter):
    def initialize(self) -> None:
        """
        Inicializa a conexão com a Bybit.
        """
        try:
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'testnet': self.testnet,
                    'accountType': 'UNIFIED',
                    'defaultContractType': 'perpetual',
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,
                    'defaultMarginMode': 'isolated'  # Sempre usar margem isolada
                }
            })
            
            # Carrega os mercados
            self.exchange.load_markets()
            self.logger.info("Bybit connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Bybit connection: {str(e)}")
            raise
            
    def get_historical_data(self, symbol: str, timeframe: str, 
                          limit: int = 100) -> pd.DataFrame:
        """
        Obtém dados históricos da Bybit.
        
        Args:
            symbol: Par de trading (ex: 'BTC/USDT')
            timeframe: Timeframe (ex: '1h', '4h', '1d')
            limit: Número de candles a retornar
            
        Returns:
            DataFrame com dados OHLCV
        """
        try:
            # Formata o símbolo para a Bybit
            formatted_symbol = self.format_symbol(symbol)
            
            # Obtém os dados
            ohlcv = self.exchange.fetch_ohlcv(
                formatted_symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Converte para DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Converte timestamp para datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise
            
    def get_current_price(self, symbol: str) -> float:
        """
        Obtém o preço atual da Bybit.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Preço atual
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            ticker = self.exchange.fetch_ticker(formatted_symbol)
            return ticker['last']
            
        except Exception as e:
            self.logger.error(f"Error fetching current price: {str(e)}")
            raise
            
    def create_order(self, symbol: str, order_type: str, side: str,
                    amount: float, price: Optional[float] = None,
                    params: Dict = None) -> Dict:
        """
        Cria uma ordem na Bybit.
        
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
        try:
            formatted_symbol = self.format_symbol(symbol)
            
            # Parâmetros padrão para a Bybit
            default_params = {
                'timeInForce': 'GTC',  # Good Till Cancel
                'reduceOnly': False,
                'closePosition': False,
                'accountType': 'UNIFIED'
            }
            
            # Mescla parâmetros padrão com parâmetros fornecidos
            if params:
                default_params.update(params)
                
            # Cria a ordem
            order = self.exchange.create_order(
                symbol=formatted_symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=default_params
            )
            
            self.logger.info(f"Order created: {order['id']}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order: {str(e)}")
            raise
            
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancela uma ordem na Bybit.
        
        Args:
            order_id: ID da ordem
            symbol: Par de trading
            
        Returns:
            Dicionário com informações da ordem cancelada
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            return self.exchange.cancel_order(order_id, formatted_symbol)
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {str(e)}")
            raise
            
    def get_balance(self, currency: str) -> Dict:
        """
        Obtém o saldo na Bybit.
        
        Args:
            currency: Símbolo da moeda
            
        Returns:
            Dicionário com informações do saldo
        """
        try:
            # Configura parâmetros específicos para a API da Bybit
            params = {
                'accountType': 'UNIFIED',
                'coin': currency
            }
            
            # Busca o saldo usando os parâmetros específicos
            balance = self.exchange.fetch_balance(params=params)
            
            # Verifica se o saldo é válido
            if not balance or currency not in balance:
                self.logger.warning(f"Currency {currency} not found in balance")
                return {
                    'free': 0.0,
                    'used': 0.0,
                    'total': 0.0
                }
            
            # Garante que os valores são números
            try:
                free = float(balance[currency]['free'] or 0.0)
                used = float(balance[currency]['used'] or 0.0)
                total = float(balance[currency]['total'] or 0.0)
            except (TypeError, ValueError):
                self.logger.warning(f"Invalid balance values for {currency}")
                return {
                    'free': 0.0,
                    'used': 0.0,
                    'total': 0.0
                }
                
            return {
                'free': free,
                'used': used,
                'total': total
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            return {
                'free': 0.0,
                'used': 0.0,
                'total': 0.0
            }
            
    def format_symbol(self, symbol: str) -> str:
        """
        Formata o símbolo para o formato da Bybit.
        
        Args:
            symbol: Par de trading no formato padrão (ex: 'BTC/USDT')
            
        Returns:
            Símbolo formatado para a Bybit (ex: 'BTCUSDT')
        """
        try:
            # Remove a barra e converte para maiúsculas
            return symbol.replace('/', '').upper()
            
        except Exception as e:
            self.logger.error(f"Error formatting symbol: {str(e)}")
            raise
            
    def set_leverage(self, symbol: str, leverage: int, 
                    buy_leverage: Optional[int] = None, 
                    sell_leverage: Optional[int] = None) -> Dict:
        """
        Define a alavancagem para um par.
        
        Args:
            symbol: Par de trading
            leverage: Nível de alavancagem padrão
            buy_leverage: Alavancagem específica para compras (opcional)
            sell_leverage: Alavancagem específica para vendas (opcional)
            
        Returns:
            Dicionário com resultado da operação
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            
            # Garante que está usando margem isolada
            self.set_margin_mode(formatted_symbol, 'isolated')
            
            # Se buy_leverage ou sell_leverage não forem especificados, usa o leverage padrão
            buy_leverage = buy_leverage or leverage
            sell_leverage = sell_leverage or leverage
            
            # Configura parâmetros específicos para a Bybit
            params = {
                'marginMode': 'isolated',
                'buyLeverage': str(buy_leverage),
                'sellLeverage': str(sell_leverage)
            }
            
            # Define a alavancagem
            return self.exchange.set_leverage(leverage, formatted_symbol, params)
            
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            raise
            
    def set_margin_mode(self, symbol: str, mode: str) -> Dict:
        """
        Define o modo de margem (isolated/cross).
        
        Args:
            symbol: Par de trading
            mode: Modo de margem ('isolated' ou 'cross')
            
        Returns:
            Dicionário com resultado da operação
        """
        try:
            formatted_symbol = self.format_symbol(symbol)
            return self.exchange.set_margin_mode(mode, formatted_symbol)
            
        except Exception as e:
            self.logger.error(f"Error setting margin mode: {str(e)}")
            raise 