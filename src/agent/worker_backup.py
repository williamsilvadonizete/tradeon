import os
import sys
import time
import logging
import asyncio
from typing import Dict, Optional, Any
import signal
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exchanges.bybit import BybitAdapter
from src.exchanges.order_manager import OrderManager
from src.risk.stop_loss_manager import StopLossManager, StopLossConfig
from src.strategies.technical_analysis import TechnicalAnalysisStrategy
from src.agent.graph import TradingAgent, MemoryGraph
from src.data.collector import DataCollector
from src.monitoring.health import HealthMonitor
from src.notifications.notifier import TelegramNotifier
from config.settings import (
    EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET,
    TRADING_PAIRS, TIMEFRAME,
    STOP_LOSS_CONFIG, WORKER_INTERVAL,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
)

class AgentState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ANALYZING = "analyzing"
    TRADING = "trading"
    ERROR = "error"
    STOPPING = "stopping"

@dataclass
class MarketEvent:
    timestamp: datetime
    symbol: str
    event_type: str
    data: Dict[str, Any]
    priority: int = 1

class IntelligentTradingAgent:
    def __init__(self):
        """
        Agente inteligente de trading que reage a eventos de mercado.
        """
        os.makedirs('logs', exist_ok=True)
        
        self.logger = self._setup_logger()
        self.state = AgentState.INACTIVE
        self.running = False
        self.event_queue = asyncio.Queue()
        
        # Componentes principais
        self.exchange = None
        self.order_manager = None
        self.strategy = None
        self.trading_agent = None
        self.memory_graph = None
        self.data_collector = None
        self.health_monitor = None
        self.notifier = None
        
        # Estado do mercado
        self.market_data = {}
        self.positions = {}
        self.last_analysis = {}
        
        # Métricas de performance
        self.metrics = {
            'trades_executed': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'uptime': 0,
            'errors': 0,
            'last_error': None
        }
        
    def _setup_logger(self) -> logging.Logger:
        """
        Configura o logger do worker.
        """
        logger = logging.getLogger('trading_worker')
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo
        fh = logging.FileHandler('logs/trading_worker.log')
        fh.setLevel(logging.INFO)
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato do log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    async def initialize_components(self):
        """
        Inicializa todos os componentes do agente.
        """
        try:
            self.logger.info("Initializing intelligent trading agent...")
            
            # Exchange e gerenciamento de ordens
            self.exchange = BybitAdapter(EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET)
            await asyncio.to_thread(self.exchange.initialize)
            
            self.order_manager = OrderManager(self.exchange)
            stop_config = StopLossConfig(**STOP_LOSS_CONFIG)
            stop_manager = StopLossManager(stop_config)
            
            # Estratégia e agente inteligente
            self.strategy = TechnicalAnalysisStrategy(self.order_manager, stop_manager)
            self.trading_agent = TradingAgent()
            self.memory_graph = MemoryGraph()
            
            # Coleta de dados e monitoramento
            self.data_collector = DataCollector(self.exchange)
            self.health_monitor = HealthMonitor()
            
            # Notificações
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            
            self.state = AgentState.ACTIVE
            self.logger.info("Agent components initialized successfully")
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics['errors'] += 1
            self.metrics['last_error'] = str(e)
            self.logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    async def process_market_event(self, event: MarketEvent):
        """
        Processa eventos de mercado de forma reativa.
        """
        try:
            self.state = AgentState.ANALYZING
            symbol = event.symbol
            
            self.logger.info(f"Processing market event for {symbol}: {event.event_type}")
            
            # Coleta dados atualizados
            market_data = await self.data_collector.get_real_time_data(symbol)
            indicators_data = await asyncio.to_thread(
                self.strategy.calculate_indicators, market_data
            )
            
            # Análise inteligente usando o agente LangGraph
            indicators_dict = self._prepare_indicators_for_analysis(indicators_data)
            decision = await asyncio.to_thread(
                self.trading_agent.analyze_market, indicators_dict
            )
            
            # Atualiza memória do agente
            self.memory_graph.store_market_state(
                symbol, indicators_dict, decision, decision
            )
            
            # Executa ação se necessário
            if decision['execute'] and decision['confidence'] > 0.7:
                await self._execute_intelligent_trade(symbol, decision, indicators_data)
            
            self.last_analysis[symbol] = {
                'timestamp': datetime.now(),
                'decision': decision,
                'event_type': event.event_type
            }
            
            self.state = AgentState.ACTIVE
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics['errors'] += 1
            self.logger.error(f"Error processing market event: {str(e)}")
    
    async def _execute_intelligent_trade(self, symbol: str, decision: Dict, data):
        """
        Executa uma operação de trading de forma inteligente.
        """
        try:
            self.state = AgentState.TRADING
            
            self.logger.info(f"Executing intelligent trade for {symbol}: {decision['action']}")
            
            # Executa a estratégia
            signal = {
                'action': decision['action'],
                'confidence': decision['confidence'],
                'reason': decision['reason']
            }
            
            result = await asyncio.to_thread(
                self.strategy.execute_trade, symbol, signal, data
            )
            
            if result:
                self.metrics['trades_executed'] += 1
                
                # Registra na memória
                self.trading_agent.update_memory({
                    'symbol': symbol,
                    'action': decision['action'],
                    'entry_price': result.get('main_order', {}).get('price', 0),
                    'confidence': decision['confidence'],
                    'reason': decision['reason']
                })
                
                # Notifica sucesso
                if self.notifier:
                    await self.notifier.send_trade_notification(symbol, result)
                
                self.logger.info(f"Trade executed successfully: {result}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            if self.notifier:
                await self.notifier.send_error_notification(str(e))
    
    def _prepare_indicators_for_analysis(self, data) -> Dict[str, Any]:
        """
        Prepara dados de indicadores para análise do agente.
        """
        last_row = data.iloc[-1]
        return {
            'rsi': float(last_row.get('rsi', 50)),
            'ema_9': float(last_row.get('ema_9', 0)),
            'ema_21': float(last_row.get('ema_21', 0)),
            'ema_50': float(last_row.get('ema_50', 0)),
            'macd': float(last_row.get('macd', 0)),
            'signal': float(last_row.get('signal', 0)),
            'upper_band': float(last_row.get('upper_band', 0)),
            'lower_band': float(last_row.get('lower_band', 0)),
            'close': float(last_row.get('close', 0)),
            'volume': float(last_row.get('volume', 0))
        }
    
    async def market_monitoring_loop(self):
        """
        Loop de monitoramento contínuo do mercado.
        """
        while self.running:
            try:
                for symbol in TRADING_PAIRS:
                    # Verifica mudanças significativas no mercado
                    current_price = await asyncio.to_thread(
                        self.exchange.get_current_price, symbol
                    )
                    
                    # Detecta eventos de mercado
                    if self._detect_market_event(symbol, current_price):
                        event = MarketEvent(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            event_type="price_change",
                            data={'price': current_price},
                            priority=1
                        )
                        await self.event_queue.put(event)
                
                # Monitoramento de saúde
                health_status = await self.health_monitor.check_system_health()
                if not health_status['healthy']:
                    self.logger.warning(f"System health issue: {health_status}")
                
                await asyncio.sleep(WORKER_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(WORKER_INTERVAL)
    
    def _detect_market_event(self, symbol: str, current_price: float) -> bool:
        """
        Detecta eventos significativos no mercado.
        """
        if symbol not in self.market_data:
            self.market_data[symbol] = {'last_price': current_price, 'last_update': datetime.now()}
            return True
        
        last_price = self.market_data[symbol]['last_price']
        price_change = abs(current_price - last_price) / last_price
        
        # Detecta mudanças > 1%
        if price_change > 0.01:
            self.market_data[symbol]['last_price'] = current_price
            self.market_data[symbol]['last_update'] = datetime.now()
            return True
        
        return False
    
    async def event_processing_loop(self):
        """
        Loop de processamento de eventos.
        """
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self.process_market_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")
    
    async def run_agent(self):
        """
        Loop principal do agente inteligente.
        """
        try:
            await self.initialize_components()
            
            self.logger.info("Starting intelligent trading agent...")
            start_time = datetime.now()
            
            # Executa loops em paralelo
            await asyncio.gather(
                self.market_monitoring_loop(),
                self.event_processing_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Fatal error in agent: {str(e)}")
            self.state = AgentState.ERROR
        finally:
            self.state = AgentState.STOPPING
            self.running = False
            
    def start(self):
        """
        Inicia o agente de trading.
        """
        if self.running:
            self.logger.warning("Agent is already running")
            return
        
        self.running = True
        self.logger.info("Starting intelligent trading agent...")
        
        # Executa o agente em um loop assíncrono
        try:
            asyncio.run(self.run_agent())
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Agent stopped due to error: {str(e)}")
        finally:
            self.running = False
    
    def stop(self):
        """
        Para o agente graciosamente.
        """
        if not self.running:
            self.logger.warning("Agent is not running")
            return
        
        self.logger.info("Stopping agent...")
        self.running = False
        self.state = AgentState.STOPPING
    
    def is_running(self) -> bool:
        """
        Verifica se o agente está rodando.
        """
        return self.running and self.state not in [AgentState.INACTIVE, AgentState.ERROR, AgentState.STOPPING]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna o status atual do agente.
        """
        return {
            'state': self.state.value,
            'running': self.running,
            'metrics': self.metrics,
            'last_analysis': self.last_analysis,
            'positions': self.positions,
            'timestamp': datetime.now().isoformat()
        }
        
# Alias para compatibilidade com código existente
TradingWorker = IntelligentTradingAgent

def signal_handler(signum, frame):
    """
    Manipulador de sinais para parada graciosa.
    """
    agent.stop()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    os.makedirs('logs', exist_ok=True)
    
    agent = IntelligentTradingAgent()
    agent.start() 