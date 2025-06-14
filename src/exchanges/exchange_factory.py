"""
Factory para criar instâncias de exchanges usando CCXT
"""

import ccxt.async_support as ccxt
from typing import Optional
from src.config.settings import (
    EXCHANGE_ID,
    EXCHANGE_API_KEY,
    EXCHANGE_SECRET,
    EXCHANGE_TESTNET
)

class ExchangeFactory:
    """Factory para criar instâncias de exchanges"""
    
    @staticmethod
    async def create_exchange() -> Optional[ccxt.Exchange]:
        """
        Cria uma instância da exchange configurada
        
        Returns:
            Optional[ccxt.Exchange]: Instância da exchange ou None em caso de erro
        """
        try:
            # Obtém a classe da exchange
            exchange_class = getattr(ccxt, EXCHANGE_ID)
            
            # Configura a exchange
            exchange = exchange_class({
                'apiKey': EXCHANGE_API_KEY,
                'secret': EXCHANGE_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'testnet': EXCHANGE_TESTNET
                }
            })
            
            # Carrega os mercados
            await exchange.load_markets()
            
            return exchange
            
        except Exception as e:
            print(f"Error creating exchange: {str(e)}")
            return None 