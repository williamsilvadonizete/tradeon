import asyncio
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import TRADING_PAIRS, TIMEFRAME, LOOKBACK_PERIOD

class DataCollector:
    """
    Coletor de dados em tempo real para o agente de trading.
    """
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = logging.getLogger('data_collector')
        self.cache = {}
        self.cache_ttl = 60  # 1 minuto
    
    async def get_real_time_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Obtém dados em tempo real para um símbolo.
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"
        now = datetime.now()
        
        # Verifica cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (now - timestamp).seconds < self.cache_ttl:
                return cached_data
        
        try:
            # Obtém dados do exchange
            data = await asyncio.to_thread(
                self.exchange.get_historical_data, symbol, timeframe, limit
            )
            
            # Atualiza cache
            self.cache[cache_key] = (data, now)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {str(e)}")
            raise
    
    async def get_market_overview(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Obtém visão geral do mercado para múltiplos símbolos.
        """
        overview = {}
        
        for symbol in symbols:
            try:
                current_price = await asyncio.to_thread(
                    self.exchange.get_current_price, symbol
                )
                
                # Calcula variação percentual (simplificado)
                data = await self.get_real_time_data(symbol, '1d', 2)
                if len(data) >= 2:
                    prev_close = data.iloc[-2]['close']
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                else:
                    change_pct = 0
                
                overview[symbol] = {
                    'price': current_price,
                    'change_24h': change_pct,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error getting overview for {symbol}: {str(e)}")
                overview[symbol] = {
                    'price': 0,
                    'change_24h': 0,
                    'error': str(e)
                }
        
        return overview
    
    def clear_cache(self):
        """
        Limpa o cache de dados.
        """
        self.cache.clear()
        self.logger.info("Data cache cleared") 