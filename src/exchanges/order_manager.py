"""
Gerenciador de ordens para exchanges
"""

from typing import Dict, Optional, List
import ccxt
import logging
from datetime import datetime
import asyncio
from ..config.settings import (
    API_KEY,
    API_SECRET,
    EXCHANGE,
    MARKET_TYPE,
    LEVERAGE,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT
)

class OrderConfig:
    """Configuração de ordem"""
    
    def __init__(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        leverage: Optional[int] = None,
        margin_mode: Optional[str] = None
    ):
        self.symbol = symbol
        self.side = side
        self.type = type
        self.amount = amount
        self.price = price
        self.stop_price = stop_price
        self.take_profit_price = take_profit_price
        self.stop_loss_price = stop_loss_price
        self.leverage = leverage
        self.margin_mode = margin_mode

class OrderManager:
    """Gerenciador de ordens"""
    
    def __init__(self):
        self.exchange = self._initialize_exchange()
        self.logger = self._setup_logger()
        self.active_orders: Dict[str, Dict] = {}
        
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Inicializa conexão com exchange"""
        try:
            exchange_class = getattr(ccxt, EXCHANGE)
            exchange = exchange_class({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True
            })
            
            if MARKET_TYPE == 'futures':
                exchange.options['defaultType'] = 'future'
                
            return exchange
            
        except Exception as e:
            raise Exception(f"Error initializing exchange: {str(e)}")
            
    def _setup_logger(self) -> logging.Logger:
        """Configura logger"""
        logger = logging.getLogger('OrderManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def create_order(self, config: OrderConfig) -> Dict:
        """
        Cria uma ordem na exchange
        
        Args:
            config: Configuração da ordem
            
        Returns:
            Dict com informações da ordem
        """
        try:
            # Configura alavancagem para futures
            if MARKET_TYPE == 'futures':
                await self._set_leverage(
                    config.symbol,
                    config.leverage or LEVERAGE
                )
                
                if config.margin_mode:
                    await self._set_margin_mode(
                        config.symbol,
                        config.margin_mode
                    )
                    
            # Cria ordem principal
            order = await self._execute_order(config)
            
            # Cria ordens de stop loss e take profit
            if config.stop_loss_price or config.take_profit_price:
                await self._create_sl_tp_orders(config, order)
                
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order: {str(e)}")
            raise
            
    async def _execute_order(self, config: OrderConfig) -> Dict:
        """Executa ordem na exchange"""
        try:
            params = {}
            
            if MARKET_TYPE == 'futures':
                params['reduceOnly'] = False
                
            order = await self.exchange.create_order(
                symbol=config.symbol,
                type=config.type,
                side=config.side,
                amount=config.amount,
                price=config.price,
                params=params
            )
            
            self.logger.info(
                f"Order created: {config.symbol} {config.side} "
                f"{config.type} {config.amount} @ {config.price}"
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            raise
            
    async def _create_sl_tp_orders(
        self,
        config: OrderConfig,
        main_order: Dict
    ) -> None:
        """Cria ordens de stop loss e take profit"""
        try:
            # Calcula preços se não fornecidos
            if not config.stop_loss_price:
                if config.side == 'buy':
                    config.stop_loss_price = main_order['price'] * (
                        1 - STOP_LOSS_PERCENT
                    )
                else:
                    config.stop_loss_price = main_order['price'] * (
                        1 + STOP_LOSS_PERCENT
                    )
                    
            if not config.take_profit_price:
                if config.side == 'buy':
                    config.take_profit_price = main_order['price'] * (
                        1 + TAKE_PROFIT_PERCENT
                    )
                else:
                    config.take_profit_price = main_order['price'] * (
                        1 - TAKE_PROFIT_PERCENT
                    )
                    
            # Cria ordem de stop loss
            sl_config = OrderConfig(
                symbol=config.symbol,
                side='sell' if config.side == 'buy' else 'buy',
                type='stop',
                amount=config.amount,
                stop_price=config.stop_loss_price,
                leverage=config.leverage,
                margin_mode=config.margin_mode
            )
            
            sl_order = await self._execute_order(sl_config)
            
            # Cria ordem de take profit
            tp_config = OrderConfig(
                symbol=config.symbol,
                side='sell' if config.side == 'buy' else 'buy',
                type='limit',
                amount=config.amount,
                price=config.take_profit_price,
                leverage=config.leverage,
                margin_mode=config.margin_mode
            )
            
            tp_order = await self._execute_order(tp_config)
            
            # Registra ordens ativas
            self.active_orders[main_order['id']] = {
                'main': main_order,
                'stop_loss': sl_order,
                'take_profit': tp_order
            }
            
        except Exception as e:
            self.logger.error(f"Error creating SL/TP orders: {str(e)}")
            raise
            
    async def _set_leverage(self, symbol: str, leverage: int) -> None:
        """Configura alavancagem"""
        try:
            await self.exchange.set_leverage(leverage, symbol)
            self.logger.info(f"Leverage set to {leverage}x for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            raise
            
    async def _set_margin_mode(self, symbol: str, mode: str) -> None:
        """Configura modo de margem"""
        try:
            await self.exchange.set_margin_mode(mode, symbol)
            self.logger.info(f"Margin mode set to {mode} for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error setting margin mode: {str(e)}")
            raise
            
    async def cancel_order(self, order_id: str, symbol: str) -> None:
        """
        Cancela uma ordem
        
        Args:
            order_id: ID da ordem
            symbol: Par de trading
        """
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order {order_id} cancelled")
            
            # Remove ordem das ordens ativas
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            raise
            
    async def get_order(self, order_id: str, symbol: str) -> Dict:
        """
        Obtém informações de uma ordem
        
        Args:
            order_id: ID da ordem
            symbol: Par de trading
            
        Returns:
            Dict com informações da ordem
        """
        try:
            return await self.exchange.fetch_order(order_id, symbol)
            
        except Exception as e:
            self.logger.error(f"Error fetching order: {str(e)}")
            raise
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtém ordens abertas
        
        Args:
            symbol: Par de trading (opcional)
            
        Returns:
            List[Dict] com ordens abertas
        """
        try:
            return await self.exchange.fetch_open_orders(symbol)
            
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {str(e)}")
            raise 