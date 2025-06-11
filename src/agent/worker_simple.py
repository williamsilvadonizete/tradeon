import os
import sys
import time
import logging
import asyncio
from typing import Dict, Optional, Any
import signal
import json
from datetime import datetime
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exchanges.bybit import BybitAdapter
from src.exchanges.order_manager import OrderManager
from src.risk.stop_loss_manager import StopLossManager, StopLossConfig
from src.strategies.technical_analysis import TechnicalAnalysisStrategy
from config.settings import (
    EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET,
    TRADING_PAIRS, TIMEFRAME,
    STOP_LOSS_CONFIG, WORKER_INTERVAL
)

class AgentState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ANALYZING = "analyzing"
    TRADING = "trading"
    ERROR = "error"
    STOPPING = "stopping"

class SimpleTradingAgent:
    def __init__(self):
        """
        Agente simplificado de trading.
        """
        os.makedirs('logs', exist_ok=True)
        
        self.logger = self._setup_logger()
        self.state = AgentState.INACTIVE
        self.running = False
        
        # Componentes principais
        self.exchange = None
        self.order_manager = None
        self.strategy = None
        
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
        Configura o logger do agente.
        """
        logger = logging.getLogger('simple_trading_agent')
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo
        fh = logging.FileHandler('logs/trading_agent.log')
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
    
    def initialize_components(self):
        """
        Inicializa todos os componentes do agente.
        """
        try:
            self.logger.info("Initializing simple trading agent...")
            
            # Exchange e gerenciamento de ordens
            if not EXCHANGE_API_KEY or not EXCHANGE_SECRET:
                self.logger.warning("API credentials not configured, using demo mode")
                return False
            
            self.exchange = BybitAdapter(EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET)
            self.exchange.initialize()
            
            self.order_manager = OrderManager(self.exchange)
            stop_config = StopLossConfig(**STOP_LOSS_CONFIG)
            stop_manager = StopLossManager(stop_config)
            
            # Estratégia
            self.strategy = TechnicalAnalysisStrategy(self.order_manager, stop_manager)
            
            self.state = AgentState.ACTIVE
            self.logger.info("Agent components initialized successfully")
            return True
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics['errors'] += 1
            self.metrics['last_error'] = str(e)
            self.logger.error(f"Error initializing agent: {str(e)}")
            return False
    
    def analyze_market(self, symbol: str):
        """
        Analisa o mercado para um símbolo específico.
        """
        try:
            self.state = AgentState.ANALYZING
            self.logger.info(f"Analyzing market for {symbol}")
            
            # Obtém dados históricos
            data = self.exchange.get_historical_data(symbol, TIMEFRAME, limit=100)
            if data is None or data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Calcula indicadores
            data = self.strategy.calculate_indicators(data)
            
            # Gera sinal
            signal = self.strategy.generate_signal(data)
            
            self.logger.info(f"Signal for {symbol}: {signal}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing market for {symbol}: {str(e)}")
            return None
    
    def execute_trade_if_needed(self, symbol: str, signal: Dict[str, Any], data):
        """
        Executa trade se o sinal for forte o suficiente.
        """
        try:
            if signal['action'] == 'hold' or signal['confidence'] < 0.7:
                return None
            
            self.state = AgentState.TRADING
            self.logger.info(f"Executing trade for {symbol}: {signal['action']}")
            
            # Executa o trade
            result = self.strategy.execute_trade(symbol, signal, data)
            
            if result:
                self.metrics['trades_executed'] += 1
                # Simular sucesso/falha (em produção, verificar resultado real)
                if signal['confidence'] > 0.8:
                    self.metrics['successful_trades'] += 1
                
                self.logger.info(f"Trade executed successfully for {symbol}")
                return result
            else:
                self.logger.warning(f"Trade execution failed for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return None
    
    def run_trading_cycle(self):
        """
        Executa um ciclo completo de trading.
        """
        try:
            self.logger.info("Starting trading cycle")
            
            for symbol in TRADING_PAIRS:
                try:
                    # Analisa o mercado
                    signal = self.analyze_market(symbol)
                    
                    if signal is None:
                        continue
                    
                    # Obtém dados para execução
                    data = self.exchange.get_historical_data(symbol, TIMEFRAME, limit=100)
                    
                    # Executa trade se necessário
                    result = self.execute_trade_if_needed(symbol, signal, data)
                    
                    if result:
                        self.logger.info(f"Trade result for {symbol}: {result}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            self.state = AgentState.ACTIVE
            self.logger.info("Trading cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")
            self.state = AgentState.ERROR
    
    def run(self):
        """
        Loop principal do agente simplificado.
        """
        try:
            # Inicializa componentes
            if not self.initialize_components():
                self.logger.error("Failed to initialize agent components")
                return
            
            self.running = True
            self.logger.info("Starting simple trading agent...")
            start_time = datetime.now()
            
            while self.running:
                try:
                    # Executa ciclo de trading
                    self.run_trading_cycle()
                    
                    # Atualiza uptime
                    self.metrics['uptime'] = (datetime.now() - start_time).total_seconds()
                    
                    # Log de status periódico
                    if self.metrics['trades_executed'] % 5 == 0 and self.metrics['trades_executed'] > 0:
                        self.logger.info(f"Agent metrics: {self.metrics}")
                    
                    # Aguarda próximo ciclo
                    time.sleep(WORKER_INTERVAL)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal, stopping...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    self.metrics['errors'] += 1
                    time.sleep(WORKER_INTERVAL)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in agent: {str(e)}")
            self.state = AgentState.ERROR
        finally:
            self.running = False
            self.state = AgentState.STOPPING
    
    def start(self):
        """
        Inicia o agente de trading.
        """
        if self.running:
            self.logger.warning("Agent is already running")
            return
        
        self.logger.info("Starting simple trading agent...")
        self.run()
    
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
            'timestamp': datetime.now().isoformat()
        }

# Alias para compatibilidade
TradingWorker = SimpleTradingAgent
IntelligentTradingAgent = SimpleTradingAgent

def signal_handler(signum, frame):
    """
    Manipulador de sinais para parada graciosa.
    """
    if 'agent' in globals():
        agent.stop()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    os.makedirs('logs', exist_ok=True)
    
    agent = SimpleTradingAgent()
    agent.start()