"""
Unified Intelligent Trading Worker
Versão única e completa com estratégias inteligentes
"""

import os
import sys
import time
import logging
import asyncio
import json
import pandas as pd
import numpy as np
import ta
import ccxt.async_support as ccxt
import openai
from typing import Dict, Optional, Any, List, TypedDict
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from src.db.database import get_db_connection
from src.db.migrations import run_migrations
from src.db.database import close_db_connection
from src.agent.exchange_manager import ExchangeManager
from src.agent.technical_indicators import TechnicalIndicators
from src.agent.simple_graph import SimpleGraph

# Configurações a partir de variáveis de ambiente
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'gateio')  # ID da exchange (gateio, binance, bybit, etc)
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY', '')
EXCHANGE_SECRET = os.getenv('EXCHANGE_SECRET', '')
EXCHANGE_TESTNET = os.getenv('EXCHANGE_TESTNET', 'false').lower() == 'true'
TRADING_PAIRS = os.getenv('TRADING_PAIRS', 'BTCUSDT,ETHUSDT').split(',')
TIMEFRAME = os.getenv('TIMEFRAME', '1h')
WORKER_INTERVAL = int(os.getenv('WORKER_INTERVAL', '60'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Parâmetros técnicos
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
EMA_FAST = int(os.getenv('EMA_FAST', '9'))
EMA_SLOW = int(os.getenv('EMA_SLOW', '21'))
MACD_FAST = int(os.getenv('MACD_FAST', '12'))
MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))
BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
BB_STD = float(os.getenv('BB_STD', '2'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

# Imports locais
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.risk.stop_loss_manager import StopLossManager
from src.strategies.technical_analysis import TechnicalAnalysisStrategy
from src.exchanges.order_manager import OrderManager

class AnalysisState(TypedDict):
    """State for the analysis workflow."""
    symbol: str
    exchange: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    indicators: Dict[str, Any]
    analysis: Dict[str, Any]
    decision: Dict[str, Any]
    confidence: float
    reasoning: str

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    rsi: float
    macd_signal: str
    ema_trend: str
    bb_position: str
    volume_surge: bool
    timestamp: datetime

class AgentState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ANALYZING = "analyzing"
    TRADING = "trading"
    ERROR = "error"
    STOPPING = "stopping"

class UnifiedIntelligentWorker:
    """Worker unificado com estratégias inteligentes"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Estado do worker
        self.state = AgentState.INACTIVE
        self.running = False
        self.cycle_count = 0
        self.pool = None
        
        # Exchange setup
        self.exchange = self.initialize_exchange()
        
        # Componentes de trading
        self.order_manager = OrderManager()
        self.stop_loss_manager = StopLossManager(self.order_manager)
        self.technical_analysis = TechnicalAnalysisStrategy(self.order_manager, self.stop_loss_manager)
        
        # OpenAI setup (with fallback)
        if OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                self.llm = ChatOpenAI(
                    model_name="gpt-4-turbo-preview",
                    temperature=0.7,
                    streaming=True
                )
                self.logger.info("✅ OpenAI configured successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ OpenAI setup failed: {e}")
                self.openai_client = None
                self.llm = None
        else:
            self.logger.warning("⚠️ OPENAI_API_KEY not configured, AI analysis disabled")
            self.openai_client = None
            self.llm = None
        
        # Análise workflow
        try:
            self.workflow = self._create_analysis_workflow()
        except Exception as e:
            self.logger.error(f"Error creating workflow: {e}")
            self.workflow = None
        
        # Trading state
        self.positions = {}
        self.last_signals = {}
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'total_pnl': 0.0,
            'trades_today': 0,
            'start_time': datetime.now()
        }
        
        # Technical Analysis Parameters
        self.ema_fast_period = int(os.getenv('EMA_FAST_PERIOD', '12'))
        self.ema_slow_period = int(os.getenv('EMA_SLOW_PERIOD', '26'))
        self.rsi_period = int(os.getenv('RSI_PERIOD', '14'))
        self.macd_fast_period = int(os.getenv('MACD_FAST_PERIOD', '12'))
        self.macd_slow_period = int(os.getenv('MACD_SLOW_PERIOD', '26'))
        self.macd_signal_period = int(os.getenv('MACD_SIGNAL_PERIOD', '9'))
        self.bb_period = int(os.getenv('BB_PERIOD', '20'))
        self.bb_std_dev = float(os.getenv('BB_STD_DEV', '2.0'))
        
        # Trading Parameters
        self.trading_pairs = os.getenv('TRADING_PAIRS', 'BTCUSDT,ETHUSDT,SOLUSDT,AAVEUSDT,SUIUSDT,BNBUSDT').split(',')
        self.timeframe = os.getenv('TIMEFRAME', '2h')
        
        # Risk Management Parameters
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '1000'))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2% risk per trade
        self.max_open_trades = int(os.getenv('MAX_OPEN_TRADES', '3'))
        
        # Performance Tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.current_balance = self.initial_capital
        
        self.logger.info(f"🤖 Unified Intelligent Worker initialized")
        self.logger.info(f"📊 Trading pairs: {TRADING_PAIRS}")
        self.logger.info(f"⏱️ Timeframe: {TIMEFRAME}, Interval: {WORKER_INTERVAL}s")
        self.logger.info(f"📈 Technical Parameters:")
        self.logger.info(f"   • RSI Period: {RSI_PERIOD}")
        self.logger.info(f"   • EMA Fast/Slow: {EMA_FAST}/{EMA_SLOW}")
        self.logger.info(f"   • MACD Fast/Slow/Signal: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
        self.logger.info(f"   • Bollinger Bands: Period={BB_PERIOD}, Std={BB_STD}")
        self.logger.info(f"   • Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    
    def initialize_exchange(self):
        """Inicializa a exchange usando CCXT"""
        try:
            # Configura a exchange
            exchange_class = getattr(ccxt, EXCHANGE_ID)
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
            
            self.logger.info(f"✅ Exchange {EXCHANGE_ID} initialized")
            self.logger.info(f"🔑 API Key: {'*' * 8}{EXCHANGE_API_KEY[-4:] if EXCHANGE_API_KEY else 'Not set'}")
            self.logger.info(f"🔒 Testnet Mode: {'Enabled' if EXCHANGE_TESTNET else 'Disabled'}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing exchange: {str(e)}")
            raise
    
    def setup_logging(self):
        """Configura logging"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/unified_worker.log')
            ]
        )
    
    async def start(self):
        """Start the worker."""
        try:
            # Initialize first
            await self.initialize()
            
            self.running = True
            self.state = AgentState.ACTIVE
            self.logger.info("🚀 Starting Unified Intelligent Worker...")
            
            while self.running:
                try:
                    # Get active exchanges and trading pairs
                    active_exchanges = self.exchange_manager.get_active_exchanges()
                    
                    for exchange in active_exchanges:
                        # Use trading_pairs directly from exchange_manager
                        for symbol in self.exchange_manager.trading_pairs:
                            await self.analyze_trading_opportunity(symbol)
                    
                    self.cycle_count += 1
                    await asyncio.sleep(60)  # Wait 1 minute between cycles
                    
                except Exception as e:
                    self.logger.error(f"Error in worker cycle: {str(e)}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Error starting worker: {str(e)}")
            raise
        finally:
            self.running = False
            self.state = AgentState.INACTIVE
            if self.pool:
                await close_db_connection()
                self.pool = None
            self.logger.info("🛑 Unified Intelligent Worker stopped")
    
    async def stop(self):
        """Para o worker"""
        self.running = False
        self.state = AgentState.INACTIVE
        
        # Fecha conexão com a exchange
        await self.exchange.close()
        
        self.logger.info("🛑 Unified Intelligent Worker stopped")
    
    async def run_trading_cycle(self):
        """Executa um ciclo completo de trading"""
        self.cycle_count += 1
        self.state = AgentState.ANALYZING
        
        self.logger.info(f"🔄 CICLO #{self.cycle_count} iniciado")
        self.logger.info(f"📊 Status atual:")
        self.logger.info(f"   • Total de sinais: {self.performance_metrics['total_signals']}")
        self.logger.info(f"   • Sinais bem-sucedidos: {self.performance_metrics['successful_signals']}")
        self.logger.info(f"   • PnL total: {self.performance_metrics['total_pnl']:.2f}")
        self.logger.info(f"   • Trades hoje: {self.performance_metrics['trades_today']}")
        
        try:
            # Analisa cada par de trading
            for i, symbol in enumerate(TRADING_PAIRS, 1):
                self.logger.info(f"📈 [{i}/{len(TRADING_PAIRS)}] Analisando {symbol}...")
                
                # Obtém dados de mercado
                ticker = await self.exchange.fetch_ticker(symbol)
                self.logger.info(f"💰 {symbol} - Preço atual: {ticker['last']:.8f}")
                self.logger.info(f"📊 {symbol} - Variação 24h: {ticker['percentage']:.2f}%")
                self.logger.info(f"📈 {symbol} - Volume 24h: {ticker['quoteVolume']:.2f}")
                
                signal = await self.analyze_symbol(symbol)
                if signal:
                    self.logger.info(f"📊 {symbol} - Análise Técnica:")
                    self.logger.info(f"   • RSI: {signal.rsi:.2f}")
                    self.logger.info(f"   • MACD: {signal.macd_signal}")
                    self.logger.info(f"   • EMA: {signal.ema_trend}")
                    self.logger.info(f"   • BB: {signal.bb_position}")
                    self.logger.info(f"   • Volume Surge: {'Sim' if signal.volume_surge else 'Não'}")
                    
                    if signal.confidence >= CONFIDENCE_THRESHOLD:
                        self.logger.info(f"🎯 SINAL DETECTADO: {symbol} {signal.action.upper()} (confiança: {signal.confidence:.2f})")
                        await self.execute_signal(signal)
                    else:
                        self.logger.info(f"⏳ {symbol}: Aguardando sinal (confiança: {signal.confidence:.2f})")
                else:
                    self.logger.info(f"⚠️ {symbol}: Sem sinal disponível")
            
            # Atualiza métricas
            self.update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")
            self.state = AgentState.ERROR
        finally:
            self.state = AgentState.ACTIVE
    
    def _create_analysis_workflow(self):
        """Create the analysis workflow using LangGraph."""
        workflow = StateGraph(AnalysisState)
        
        # Define nodes
        workflow.add_node("analyze_indicators", self._analyze_indicators)
        workflow.add_node("analyze_market", self._analyze_market)
        workflow.add_node("make_decision", self._make_decision)
        
        # Define edges
        workflow.add_edge("analyze_indicators", "analyze_market")
        workflow.add_edge("analyze_market", "make_decision")
        workflow.add_edge("make_decision", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_indicators")
        
        return workflow.compile()
    
    async def _analyze_indicators(self, state: AnalysisState) -> AnalysisState:
        """Analyze technical indicators."""
        try:
            # Get historical data
            candles = await self.exchange.fetch_ohlcv(
                state["symbol"],
                timeframe=TIMEFRAME,
                limit=100
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
            df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=EMA_FAST).ema_indicator()
            df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=EMA_SLOW).ema_indicator()
            
            macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            bb = ta.volatility.BollingerBands(df['close'], window=BB_PERIOD, window_dev=BB_STD)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            
            # Get current values
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd_line = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            bb_high = df['bb_high'].iloc[-1]
            bb_low = df['bb_low'].iloc[-1]
            
            # Calculate price change and volume
            price_change_24h = (current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            volume_24h = df['volume'].sum()
            
            state.update({
                "current_price": current_price,
                "price_change_24h": price_change_24h,
                "volume_24h": volume_24h,
                "indicators": {
                    "rsi": rsi,
                    "macd": {
                        "line": macd_line,
                        "signal": macd_signal
                    },
                    "bollinger_bands": {
                        "high": bb_high,
                        "low": bb_low
                    },
                    "ema": {
                        "fast": df['ema_fast'].iloc[-1],
                        "slow": df['ema_slow'].iloc[-1]
                    }
                }
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error analyzing indicators: {str(e)}")
            raise
    
    async def _analyze_market(self, state: AnalysisState) -> AnalysisState:
        """Analyze market conditions using OpenAI."""
        try:
            if not self.llm:
                self.logger.warning("OpenAI not available, skipping market analysis")
                state["analysis"] = {
                    "market_analysis": "AI analysis not available",
                    "timestamp": datetime.now().isoformat()
                }
                return state
            # Prepare market analysis prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert crypto market analyst. 
                Analyze the following market data and provide insights about market conditions, 
                trends, and potential trading opportunities."""),
                HumanMessage(content=f"""Analyze the following market data for {state['symbol']}:
                
                Current Price: ${state['current_price']:,.2f}
                24h Change: {state['price_change_24h']:.2f}%
                24h Volume: {state['volume_24h']:,.2f}
                
                Technical Indicators:
                RSI: {state['indicators']['rsi']:.2f}
                MACD: {state['indicators']['macd']['line']:.2f}
                MACD Signal: {state['indicators']['macd']['signal']:.2f}
                Bollinger Bands:
                - Upper: {state['indicators']['bollinger_bands']['high']:.2f}
                - Lower: {state['indicators']['bollinger_bands']['low']:.2f}
                
                Please provide:
                1. Market trend analysis
                2. Key support/resistance levels
                3. Volume analysis
                4. Risk assessment
                5. Trading opportunity assessment""")
            ])
            
            # Get analysis from OpenAI
            response = await self.llm.ainvoke(prompt)
            analysis = response.content
            
            state["analysis"] = {
                "market_analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            raise
    
    async def _make_decision(self, state: AnalysisState) -> AnalysisState:
        """Make trading decision based on analysis."""
        try:
            if not self.llm:
                self.logger.warning("OpenAI not available, using basic decision logic")
                # Basic decision based on RSI
                rsi = state["indicators"]["rsi"]
                if rsi < 30:
                    action = "buy"
                    confidence = 0.6
                elif rsi > 70:
                    action = "sell"
                    confidence = 0.6
                else:
                    action = "hold"
                    confidence = 0.3
                
                state["decision"] = {
                    "action": action,
                    "confidence": confidence,
                    "reasoning": f"Basic RSI-based decision: RSI={rsi:.2f}",
                    "timestamp": datetime.now().isoformat()
                }
                return state
            # Prepare decision prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert crypto trader. 
                Based on the market analysis, make a trading decision (buy, sell, or hold) 
                and provide your confidence level and reasoning."""),
                HumanMessage(content=f"""Based on the following analysis for {state['symbol']}, 
                make a trading decision:
                
                {state['analysis']['market_analysis']}
                
                Please provide:
                1. Trading decision (buy/sell/hold)
                2. Confidence level (0-100%)
                3. Detailed reasoning
                4. Risk management recommendations""")
            ])
            
            # Get decision from OpenAI
            response = await self.llm.ainvoke(prompt)
            decision = response.content
            
            # Parse decision and confidence
            decision_lines = decision.split('\n')
            action = None
            confidence = 0.0
            
            for line in decision_lines:
                if 'decision' in line.lower():
                    if 'buy' in line.lower():
                        action = 'buy'
                    elif 'sell' in line.lower():
                        action = 'sell'
                    else:
                        action = 'hold'
                elif 'confidence' in line.lower():
                    try:
                        confidence = float(line.split('%')[0].split()[-1]) / 100
                    except:
                        confidence = 0.5
            
            state["decision"] = {
                "action": action or 'hold',
                "confidence": confidence,
                "reasoning": decision,
                "timestamp": datetime.now().isoformat()
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error making decision: {str(e)}")
            raise

    async def analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Analisa um símbolo e gera sinal de trading"""
        try:
            # Initialize analysis state
            state = AnalysisState(
                symbol=symbol,
                exchange=EXCHANGE_ID,
                current_price=0.0,
                price_change_24h=0.0,
                volume_24h=0.0,
                indicators={},
                analysis={},
                decision={},
                confidence=0.0,
                reasoning=""
            )
            
            # Run analysis workflow
            final_state = await self.workflow.ainvoke(state)
            
            # Create trading signal
            if final_state["decision"]["confidence"] >= CONFIDENCE_THRESHOLD:
                signal = TradingSignal(
                    symbol=symbol,
                    action=final_state["decision"]["action"],
                    confidence=final_state["decision"]["confidence"],
                    price=final_state["current_price"],
                    rsi=final_state["indicators"]["rsi"],
                    macd_signal="bullish" if final_state["indicators"]["macd"]["line"] > final_state["indicators"]["macd"]["signal"] else "bearish",
                    ema_trend="bullish" if final_state["indicators"]["ema"]["fast"] > final_state["indicators"]["ema"]["slow"] else "bearish",
                    bb_position="overbought" if final_state["current_price"] > final_state["indicators"]["bollinger_bands"]["high"] else "oversold" if final_state["current_price"] < final_state["indicators"]["bollinger_bands"]["low"] else "neutral",
                    volume_surge=final_state["volume_24h"] > final_state["indicators"]["volume_24h"] * 1.5,
                    timestamp=datetime.now()
                )
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
            return None
    
    async def execute_signal(self, signal: TradingSignal):
        """Executa um sinal de trading"""
        try:
            if signal.action == "hold":
                return
            
            self.state = AgentState.TRADING
            
            # Verifica se já tem posição
            if signal.symbol in self.positions:
                if (self.positions[signal.symbol]['side'] == 'long' and signal.action == 'buy') or \
                   (self.positions[signal.symbol]['side'] == 'short' and signal.action == 'sell'):
                    self.logger.info(f"Already have position in {signal.symbol}, skipping")
                    return
            
            # Calcula tamanho da posição
            amount = self.calculate_position_size(signal)
            
            # Executa ordem
            order = await self.exchange.create_order(
                symbol=signal.symbol,
                type='market',
                side=signal.action,
                amount=amount
            )
            
            if order:
                self.positions[signal.symbol] = {
                    'side': 'long' if signal.action == 'buy' else 'short',
                    'entry_price': signal.price,
                    'timestamp': datetime.now()
                }
                
                self.performance_metrics['total_signals'] += 1
                self.logger.info(f"Order executed: {order}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {str(e)}")
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """Calcula tamanho da posição baseado em risco"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            risk_per_trade = 0.02  # 2% do capital por trade
            
            position_size = (usdt_balance * risk_per_trade) / signal.price
            return round(position_size, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def update_performance_metrics(self):
        """Atualiza métricas de performance"""
        try:
            # Calcula PnL total
            total_pnl = 0.0
            for symbol, position in self.positions.items():
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                if position['side'] == 'long':
                    pnl = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl = (position['entry_price'] - current_price) / position['entry_price']
                total_pnl += pnl
            
            self.performance_metrics['total_pnl'] = total_pnl
            
            # Log de métricas
            self.logger.info(f"Performance metrics: {self.performance_metrics}")
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    async def initialize(self):
        """Initialize the worker."""
        try:
            self.logger.info("🔄 Initializing worker...")
            
            # Initialize database connection
            self.logger.info("📊 Initializing database connection...")
            self.pool = await get_db_connection()
            if not self.pool:
                raise Exception("Failed to initialize database connection pool")
            
            # Run database migrations
            self.logger.info("🔄 Running database migrations...")
            success = await run_migrations()
            if not success:
                raise Exception("Failed to run database migrations")
            
            # Initialize exchange manager
            self.logger.info("💱 Initializing exchange manager...")
            self.exchange_manager = ExchangeManager()
            await self.exchange_manager.initialize()
            
            # Initialize technical indicators
            self.logger.info("📈 Initializing technical indicators...")
            self.technical_indicators = TechnicalIndicators()
            
            # Initialize knowledge graph
            self.logger.info("🧠 Initializing knowledge graph...")
            self.knowledge_graph = SimpleGraph()
            
            self.logger.info("✅ Worker initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing worker: {str(e)}")
            raise

    async def log_operation(self, symbol: str, exchange_id: str, analysis: Dict):
        """Log operation to database."""
        try:
            if not self.pool:
                self.logger.warning("Database connection pool not initialized")
                return
                
            # Adiciona informações detalhadas ao log
            current_price = analysis['indicators']['current_price']
            rsi = analysis['indicators']['rsi']
            macd_line = analysis['indicators']['macd_line']
            macd_signal = analysis['indicators']['macd_signal']
            bb_upper = analysis['indicators']['bb_upper']
            bb_lower = analysis['indicators']['bb_lower']
            
            # Formata o log detalhado
            detailed_analysis = {
                'price': current_price,
                'indicators': {
                    'rsi': rsi,
                    'macd': {
                        'line': macd_line,
                        'signal': macd_signal
                    },
                    'bollinger_bands': {
                        'upper': bb_upper,
                        'lower': bb_lower
                    }
                },
                'evaluation': analysis['evaluation'],
                'score': analysis['score'],
                'reasons': analysis['reasons']
            }
            
            # Log no console
            self.logger.info(f"\n=== {symbol} Analysis ===")
            self.logger.info(f"Current Price: ${current_price:,.2f}")
            self.logger.info(f"RSI: {rsi:.2f}")
            self.logger.info(f"MACD: {macd_line:.2f} (Signal: {macd_signal:.2f})")
            self.logger.info(f"Bollinger Bands: Upper ${bb_upper:,.2f} | Lower ${bb_lower:,.2f}")
            self.logger.info(f"Evaluation: {analysis['evaluation']}")
            self.logger.info(f"Score: {analysis['score']}")
            self.logger.info(f"Reasons: {', '.join(analysis['reasons'])}")
            self.logger.info("=== End Analysis ===\n")
            
            # Salva no banco de dados
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO worker_logs (symbol, exchange_id, analysis, created_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                """, symbol, exchange_id, json.dumps(detailed_analysis))
                self.logger.info(f"Operation logged for {symbol} on {exchange_id}")
        except Exception as e:
            self.logger.error(f"Error logging operation: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error details: {str(e)}")

    async def stop(self):
        """Stop the worker."""
        self.running = False
        self.state = AgentState.INACTIVE
        
        # Fecha conexão com a exchange
        await self.exchange.close()
        
        if self.pool:
            await close_db_connection()
            self.pool = None
            self.logger.info("Database connection pool closed")
        
        self.logger.info("🛑 Unified Intelligent Worker stopped")

    async def analyze_trading_opportunity(self, symbol: str) -> None:
        """Analyze trading opportunity for a symbol."""
        try:
            self.logger.info(f"Analyzing trading opportunity for {symbol}")
            
            # Fetch historical data using timeframe from environment variable
            df = await self.exchange_manager.fetch_historical_data(symbol, self.timeframe)
            if df is None:
                self.logger.warning(f"No historical data available for {symbol}")
                return
            
            # Calculate technical indicators
            df['ema_fast'] = self.technical_indicators.calculate_ema(df['close'], self.ema_fast_period)
            df['ema_slow'] = self.technical_indicators.calculate_ema(df['close'], self.ema_slow_period)
            df['rsi'] = self.technical_indicators.calculate_rsi(df['close'], self.rsi_period)
            
            # Calculate MACD
            macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(
                df['close'],
                self.macd_fast_period,
                self.macd_slow_period,
                self.macd_signal_period
            )
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = histogram
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.technical_indicators.calculate_bollinger_bands(
                df['close'],
                self.bb_period,
                self.bb_std_dev
            )
            df['bb_upper'] = upper_band
            df['bb_middle'] = middle_band
            df['bb_lower'] = lower_band
            
            # Get the latest values
            latest = df.iloc[-1]
            
            # Log the analysis
            self.logger.info(f"Analysis for {symbol}:")
            self.logger.info(f"Current Price: {latest['close']:.2f}")
            self.logger.info(f"RSI: {latest['rsi']:.2f}")
            self.logger.info(f"MACD: {latest['macd']:.2f}")
            self.logger.info(f"MACD Signal: {latest['macd_signal']:.2f}")
            self.logger.info(f"MACD Histogram: {latest['macd_hist']:.2f}")
            self.logger.info(f"BB Upper: {latest['bb_upper']:.2f}")
            self.logger.info(f"BB Middle: {latest['bb_middle']:.2f}")
            self.logger.info(f"BB Lower: {latest['bb_lower']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing trading opportunity for {symbol}: {str(e)}")
            raise

if __name__ == "__main__":
    worker = UnifiedIntelligentWorker()
    asyncio.run(worker.start())