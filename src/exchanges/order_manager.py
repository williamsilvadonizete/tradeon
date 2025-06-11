from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OrderConfig:
    """Configuração para criação de ordens"""
    symbol: str
    side: str  # 'buy' ou 'sell'
    order_type: str  # 'limit' ou 'market'
    amount: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_stop_activation: Optional[float] = None
    reduce_only: bool = False
    close_position: bool = False

class OrderManager:
    def __init__(self, exchange_adapter):
        """
        Inicializa o gerenciador de ordens.
        
        Args:
            exchange_adapter: Adaptador da exchange
        """
        self.exchange = exchange_adapter
        self.logger = logging.getLogger(__name__)
        self.active_stops = {}  # Armazena stops ativos
        
    def create_order_with_stops(self, config: OrderConfig) -> Tuple[Dict, Dict, Dict]:
        """
        Cria uma ordem principal com stop loss e take profit.
        
        Args:
            config: Configuração da ordem
            
        Returns:
            Tuple contendo (ordem principal, stop loss, take profit)
        """
        try:
            # 1. Cria a ordem principal
            main_order = self.exchange.create_order(
                symbol=config.symbol,
                order_type=config.order_type,
                side=config.side,
                amount=config.amount,
                price=config.price
            )
            
            self.logger.info(f"Main order created: {main_order['id']}")
            
            # 2. Cria o stop loss se especificado
            stop_loss_order = None
            if config.stop_loss:
                stop_loss_order = self._create_stop_loss(
                    config.symbol,
                    config.amount,
                    config.stop_loss,
                    config.side
                )
                
            # 3. Cria o take profit se especificado
            take_profit_order = None
            if config.take_profit:
                take_profit_order = self._create_take_profit(
                    config.symbol,
                    config.amount,
                    config.take_profit,
                    config.side
                )
                
            # 4. Configura trailing stop se especificado
            if config.trailing_stop:
                self._setup_trailing_stop(
                    config.symbol,
                    config.amount,
                    config.trailing_stop,
                    config.trailing_stop_activation,
                    config.side
                )
                
            return main_order, stop_loss_order, take_profit_order
            
        except Exception as e:
            self.logger.error(f"Error creating order with stops: {str(e)}")
            # Tenta cancelar ordens parciais em caso de erro
            self._cleanup_partial_orders(main_order, stop_loss_order, take_profit_order)
            raise
            
    def _create_stop_loss(self, symbol: str, amount: float, 
                         stop_price: float, side: str) -> Dict:
        """
        Cria uma ordem de stop loss.
        
        Args:
            symbol: Par de trading
            amount: Quantidade
            stop_price: Preço do stop
            side: Lado da ordem original
            
        Returns:
            Ordem de stop loss criada
        """
        try:
            # Inverte o lado para o stop loss
            stop_side = 'sell' if side == 'buy' else 'buy'
            
            stop_order = self.exchange.create_order(
                symbol=symbol,
                order_type='stop_market',
                side=stop_side,
                amount=amount,
                params={
                    'stopPrice': stop_price,
                    'reduceOnly': True
                }
            )
            
            self.logger.info(f"Stop loss created: {stop_order['id']}")
            return stop_order
            
        except Exception as e:
            self.logger.error(f"Error creating stop loss: {str(e)}")
            raise
            
    def _create_take_profit(self, symbol: str, amount: float,
                          take_profit_price: float, side: str) -> Dict:
        """
        Cria uma ordem de take profit.
        
        Args:
            symbol: Par de trading
            amount: Quantidade
            take_profit_price: Preço do take profit
            side: Lado da ordem original
            
        Returns:
            Ordem de take profit criada
        """
        try:
            # Inverte o lado para o take profit
            tp_side = 'sell' if side == 'buy' else 'buy'
            
            tp_order = self.exchange.create_order(
                symbol=symbol,
                order_type='limit',
                side=tp_side,
                amount=amount,
                price=take_profit_price,
                params={'reduceOnly': True}
            )
            
            self.logger.info(f"Take profit created: {tp_order['id']}")
            return tp_order
            
        except Exception as e:
            self.logger.error(f"Error creating take profit: {str(e)}")
            raise
            
    def _setup_trailing_stop(self, symbol: str, amount: float,
                           trailing_stop: float, activation_price: float,
                           side: str) -> None:
        """
        Configura um trailing stop.
        
        Args:
            symbol: Par de trading
            amount: Quantidade
            trailing_stop: Distância do trailing stop
            activation_price: Preço de ativação
            side: Lado da ordem original
        """
        try:
            # Inverte o lado para o trailing stop
            trailing_side = 'sell' if side == 'buy' else 'buy'
            
            trailing_order = self.exchange.create_order(
                symbol=symbol,
                order_type='trailing_stop',
                side=trailing_side,
                amount=amount,
                params={
                    'trailingStop': trailing_stop,
                    'activationPrice': activation_price,
                    'reduceOnly': True
                }
            )
            
            self.logger.info(f"Trailing stop created: {trailing_order['id']}")
            self.active_stops[trailing_order['id']] = {
                'symbol': symbol,
                'amount': amount,
                'side': trailing_side,
                'created_at': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up trailing stop: {str(e)}")
            raise
            
    def _cleanup_partial_orders(self, main_order: Optional[Dict],
                              stop_loss: Optional[Dict],
                              take_profit: Optional[Dict]) -> None:
        """
        Limpa ordens parciais em caso de erro.
        
        Args:
            main_order: Ordem principal
            stop_loss: Ordem de stop loss
            take_profit: Ordem de take profit
        """
        try:
            if main_order:
                self.exchange.cancel_order(main_order['id'], main_order['symbol'])
            if stop_loss:
                self.exchange.cancel_order(stop_loss['id'], stop_loss['symbol'])
            if take_profit:
                self.exchange.cancel_order(take_profit['id'], take_profit['symbol'])
        except Exception as e:
            self.logger.error(f"Error cleaning up partial orders: {str(e)}")
            
    def update_stop_loss(self, order_id: str, new_stop_price: float) -> Dict:
        """
        Atualiza o preço de um stop loss existente.
        
        Args:
            order_id: ID da ordem de stop loss
            new_stop_price: Novo preço do stop
            
        Returns:
            Ordem atualizada
        """
        try:
            # Cancela o stop loss antigo
            old_order = self.exchange.fetch_order(order_id)
            self.exchange.cancel_order(order_id, old_order['symbol'])
            
            # Cria um novo stop loss
            new_stop = self._create_stop_loss(
                old_order['symbol'],
                old_order['amount'],
                new_stop_price,
                'buy' if old_order['side'] == 'sell' else 'sell'
            )
            
            self.logger.info(f"Stop loss updated: {new_stop['id']}")
            return new_stop
            
        except Exception as e:
            self.logger.error(f"Error updating stop loss: {str(e)}")
            raise
            
    def cancel_all_stops(self, symbol: str) -> None:
        """
        Cancela todos os stops ativos para um símbolo.
        
        Args:
            symbol: Par de trading
        """
        try:
            for stop_id, stop_info in self.active_stops.items():
                if stop_info['symbol'] == symbol:
                    self.exchange.cancel_order(stop_id, symbol)
                    del self.active_stops[stop_id]
                    
            self.logger.info(f"All stops cancelled for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error cancelling stops: {str(e)}")
            raise 