import os
import sys
import time
import logging
import asyncio
from typing import Dict, Optional, Any, List
import signal
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import structlog

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exchanges.bybit import BybitAdapter
from src.exchanges.mexc import MEXCAdapter
from src.exchanges.gateio import GateIOAdapter
from src.exchanges.order_manager import OrderManager
from src.risk.stop_loss_manager import StopLossManager, StopLossConfig
from src.strategies.technical_analysis import TechnicalAnalysisStrategy
from config.settings import (
    EXCHANGE_ID, EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET,
    TRADING_PAIRS, TIMEFRAME,
    STOP_LOSS_CONFIG, WORKER_INTERVAL
)

# Optional imports - graceful fallback if not available
HAS_ADVANCED_FEATURES = True
try:
    from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = None
    TELEGRAM_CHAT_ID = None
    HAS_ADVANCED_FEATURES = False

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

class RobustTradingAgent:
    def __init__(self):
        """
        Agente robusto de trading com fallbacks.
        """
        os.makedirs('logs', exist_ok=True)
        
        self.logger = self._setup_logger()
        self.state = AgentState.INACTIVE
        self.running = False
        self.event_queue = None
        
        # Componentes principais
        self.exchange = None
        self.order_manager = None
        self.strategy = None
        
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
        Configura o logger estruturado do worker com logs detalhados.
        """
        # Configurar estrutlog para logs estruturados
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        logger = logging.getLogger('crypto_trading_worker')
        logger.setLevel(logging.INFO)
        
        # Remove handlers existentes
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Handler para arquivo com logs detalhados
        fh = logging.FileHandler('logs/trading_worker.log')
        fh.setLevel(logging.INFO)
        
        # Handler para console com formato mais legível
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato detalhado para arquivo (JSON estruturado)
        file_formatter = logging.Formatter('%(message)s')
        
        # Formato legível para console
        console_formatter = logging.Formatter(
            '🤖 %(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def _create_exchange_adapter(self):
        """
        Factory method para criar o adapter de exchange baseado no EXCHANGE_ID.
        """
        exchange_adapters = {
            'bybit': BybitAdapter,
            'mexc': MEXCAdapter,
            'gateio': GateIOAdapter,
            'gate': GateIOAdapter  # Alias para gate.io
        }
        
        adapter_class = exchange_adapters.get(EXCHANGE_ID.lower())
        if not adapter_class:
            raise ValueError(f"Exchange '{EXCHANGE_ID}' não suportado. Exchanges disponíveis: {list(exchange_adapters.keys())}")
        
        self.logger.info(f"Criando adapter para exchange: {EXCHANGE_ID}")
        return adapter_class(EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET)

    def log_market_analysis(self, symbol: str, price: float, analysis: Dict[str, Any], decision: str):
        """
        Log detalhado da análise de mercado.
        """
        analysis_data = {
            "event": "market_analysis",
            "symbol": symbol,
            "current_price": price,
            "technical_indicators": analysis.get('indicators', {}),
            "signal_strength": analysis.get('signal_strength', 0),
            "decision": decision,
            "reason": analysis.get('reason', 'Análise técnica'),
            "timestamp": datetime.now().isoformat()
        }
        
        # Log estruturado para arquivo
        self.logger.info(json.dumps(analysis_data))
        
        # Log legível para console
        decision_emoji = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "⚫"
        self.logger.info(
            f"{decision_emoji} ANÁLISE {symbol}: Preço=${price:.4f} | "
            f"Decisão={decision} | Força={analysis.get('signal_strength', 0):.2f} | "
            f"Razão: {analysis.get('reason', 'N/A')}"
        )

    def log_trade_execution(self, symbol: str, action: str, amount: float, price: float, success: bool, reason: str = ""):
        """
        Log detalhado da execução de trades.
        """
        trade_data = {
            "event": "trade_execution",
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "price": price,
            "success": success,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "value_usd": amount * price
        }
        
        # Log estruturado para arquivo
        self.logger.info(json.dumps(trade_data))
        
        # Log legível para console
        status_emoji = "✅" if success else "❌"
        value_usd = amount * price
        self.logger.info(
            f"{status_emoji} TRADE {action} {symbol}: "
            f"Quantidade={amount:.6f} | Preço=${price:.4f} | "
            f"Valor=${value_usd:.2f} | Status={'Sucesso' if success else 'Falha'}"
            f"{' | Razão: ' + reason if reason else ''}"
        )

    def log_strategy_decision(self, symbol: str, indicators: Dict[str, float], decision_factors: List[str], final_decision: str):
        """
        Log detalhado das decisões estratégicas.
        """
        strategy_data = {
            "event": "strategy_decision",
            "symbol": symbol,
            "indicators": indicators,
            "decision_factors": decision_factors,
            "final_decision": final_decision,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log estruturado para arquivo
        self.logger.info(json.dumps(strategy_data))
        
        # Log legível para console
        factors_str = " | ".join(decision_factors)
        self.logger.info(
            f"📊 ESTRATÉGIA {symbol}: {final_decision} | "
            f"Indicadores: RSI={indicators.get('rsi', 0):.2f}, "
            f"MACD={indicators.get('macd', 0):.4f} | "
            f"Fatores: {factors_str}"
        )

    def log_portfolio_status(self, total_value: float, positions: Dict[str, Any], pnl_24h: float):
        """
        Log do status do portfólio.
        """
        portfolio_data = {
            "event": "portfolio_status",
            "total_value_usd": total_value,
            "positions": positions,
            "pnl_24h": pnl_24h,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log estruturado para arquivo
        self.logger.info(json.dumps(portfolio_data))
        
        # Log legível para console
        pnl_emoji = "📈" if pnl_24h > 0 else "📉" if pnl_24h < 0 else "➖"
        positions_count = len([p for p in positions.values() if p.get('size', 0) > 0])
        self.logger.info(
            f"💼 PORTFÓLIO: Valor=${total_value:.2f} | "
            f"Posições={positions_count} | "
            f"P&L 24h={pnl_emoji}${pnl_24h:.2f}"
        )

    def log_risk_management(self, symbol: str, action: str, current_risk: float, max_risk: float, details: str):
        """
        Log das decisões de gestão de risco.
        """
        risk_data = {
            "event": "risk_management",
            "symbol": symbol,
            "action": action,
            "current_risk_pct": current_risk,
            "max_risk_pct": max_risk,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log estruturado para arquivo
        self.logger.info(json.dumps(risk_data))
        
        # Log legível para console
        risk_emoji = "🛡️" if action == "PROTECT" else "⚠️" if action == "WARNING" else "🚨"
        self.logger.info(
            f"{risk_emoji} RISCO {symbol}: {action} | "
            f"Risco Atual={current_risk:.2f}% | Max={max_risk:.2f}% | "
            f"Detalhes: {details}"
        )

    def initialize_components(self):
        """
        Inicializa todos os componentes do agente.
        """
        try:
            self.logger.info("Initializing robust trading agent...")
            
            # TESTE DE CONECTIVIDADE GATE.IO
            self.logger.info("🧪 Running Gate.io connectivity test...")
            try:
                # Importa e executa as funções de teste diretamente
                sys.path.append('/app/src')
                from test_gateio_connection import test_basic_connectivity, test_gateio_public_api, test_gateio_connection
                
                # 1. Teste básico de conectividade
                self.logger.info("🌐 Testing basic internet connectivity...")
                connectivity_ok = test_basic_connectivity()
                
                if connectivity_ok:
                    self.logger.info("✅ Internet connectivity OK")
                    
                    # 2. Teste API pública Gate.io
                    self.logger.info("🌐 Testing Gate.io public API...")
                    public_api_ok = test_gateio_public_api()
                    
                    if public_api_ok:
                        self.logger.info("✅ Gate.io public API OK")
                        
                        # 3. Teste API privada (com credenciais)
                        self.logger.info("🔑 Testing Gate.io private API with credentials...")
                        auth_ok = test_gateio_connection()
                        
                        if auth_ok:
                            self.logger.info("🎉 ALL TESTS PASSED! Gate.io is working")
                        else:
                            self.logger.error("❌ Gate.io authentication failed")
                    else:
                        self.logger.error("❌ Gate.io public API not accessible")
                else:
                    self.logger.error("❌ No internet connectivity")
                            
            except Exception as test_error:
                self.logger.error(f"❌ Error running connectivity test: {str(test_error)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Verifica credenciais
            if not EXCHANGE_API_KEY or not EXCHANGE_SECRET:
                self.logger.warning("Exchange credentials not found, running in demo mode")
                self.state = AgentState.ACTIVE
                return True
            
            # Exchange e gerenciamento de ordens
            self.exchange = self._create_exchange_adapter()
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
            self.logger.error(f"Error initializing agent: {str(e)}")
            self.logger.warning("Exchange connection failed, switching to demo mode for analysis")
            
            # Modo demo - sem exchange real, mas com análises simuladas
            self.exchange = None
            self.order_manager = None
            self.strategy = None
            
            self.state = AgentState.ACTIVE
            self.metrics['errors'] += 1
            self.metrics['last_error'] = str(e)
            
            self.logger.info("✅ Demo mode initialized - will perform simulated analysis")
            return True

    def analyze_and_trade(self, symbol: str):
        """
        Analisa mercado e executa trades para um símbolo com logs detalhados.
        """
        try:
            self.state = AgentState.ANALYZING
            self.logger.info(f"🔍 Iniciando análise completa para {symbol}")
            
            if not self.exchange:
                self.logger.info(f"🎯 Executando análise simulada para {symbol} (modo demo)")
                self._perform_demo_analysis(symbol)
                return
            
            # Obtém dados históricos
            data = self.exchange.get_historical_data(symbol, TIMEFRAME, limit=100)
            if data is None or data.empty:
                self.logger.warning(f"⚠️ Dados históricos não disponíveis para {symbol}")
                return
            
            current_price = float(data['close'].iloc[-1])
            price_change_24h = ((current_price - float(data['close'].iloc[-24])) / float(data['close'].iloc[-24])) * 100 if len(data) >= 24 else 0
            
            self.logger.info(f"📊 {symbol}: Preço=${current_price:.4f} | Variação 24h={price_change_24h:+.2f}%")
            
            # Calcula indicadores técnicos
            data = self.strategy.calculate_indicators(data)
            
            # Extrai indicadores principais para análise
            indicators = {}
            decision_factors = []
            
            if 'rsi' in data.columns:
                rsi = float(data['rsi'].iloc[-1])
                indicators['rsi'] = rsi
                if rsi > 70:
                    decision_factors.append(f"RSI sobrecomprado ({rsi:.1f})")
                elif rsi < 30:
                    decision_factors.append(f"RSI sobrevendido ({rsi:.1f})")
                else:
                    decision_factors.append(f"RSI neutro ({rsi:.1f})")
            
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = float(data['macd'].iloc[-1])
                macd_signal = float(data['macd_signal'].iloc[-1])
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                
                if macd > macd_signal:
                    decision_factors.append("MACD bullish (cruzamento positivo)")
                else:
                    decision_factors.append("MACD bearish (cruzamento negativo)")
            
            if 'ema_fast' in data.columns and 'ema_slow' in data.columns:
                ema_fast = float(data['ema_fast'].iloc[-1])
                ema_slow = float(data['ema_slow'].iloc[-1])
                indicators['ema_fast'] = ema_fast
                indicators['ema_slow'] = ema_slow
                
                if ema_fast > ema_slow:
                    decision_factors.append("Tendência de alta (EMA rápida > lenta)")
                else:
                    decision_factors.append("Tendência de baixa (EMA rápida < lenta)")
            
            # Gera sinal de trading
            signal = self.strategy.generate_signal(data)
            
            # Log da estratégia detalhada
            self.log_strategy_decision(symbol, indicators, decision_factors, signal['action'].upper())
            
            # Análise de volume (se disponível)
            volume_analysis = ""
            if 'volume' in data.columns:
                current_volume = float(data['volume'].iloc[-1])
                avg_volume = float(data['volume'].tail(20).mean())
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 1.5:
                    volume_analysis = f"Volume alto ({volume_ratio:.1f}x média)"
                elif volume_ratio < 0.5:
                    volume_analysis = f"Volume baixo ({volume_ratio:.1f}x média)"
                else:
                    volume_analysis = f"Volume normal ({volume_ratio:.1f}x média)"
                
                decision_factors.append(volume_analysis)
            
            # Log da análise de mercado completa
            analysis_data = {
                'indicators': indicators,
                'signal_strength': signal['confidence'],
                'volume_analysis': volume_analysis,
                'price_change_24h': price_change_24h,
                'reason': f"Análise baseada em: {', '.join(decision_factors)}"
            }
            self.log_market_analysis(symbol, current_price, analysis_data, signal['action'].upper())
            
            # Calcula risco do portfólio
            current_risk = self._calculate_portfolio_risk()
            max_risk = 10.0  # 10% máximo
            
            # Armazena análise detalhada
            self.last_analysis[symbol] = {
                'timestamp': datetime.now(),
                'signal': signal,
                'indicators': indicators,
                'price': current_price,
                'price_change_24h': price_change_24h,
                'decision_factors': decision_factors,
                'volume_analysis': volume_analysis
            }
            
            # Decisão de execução com gestão de risco
            if current_risk > max_risk:
                self.log_risk_management(
                    symbol, "BLOCK", current_risk, max_risk,
                    f"Bloqueando trades - risco do portfólio muito alto"
                )
                return
            
            # Executa trade se sinal for forte o suficiente
            if signal['action'] != 'hold' and signal['confidence'] > 0.7:
                self.log_risk_management(
                    symbol, "APPROVE", current_risk, max_risk,
                    f"Trade aprovado - risco dentro do limite (confiança: {signal['confidence']:.2f})"
                )
                self._execute_trade(symbol, signal, data)
            elif signal['action'] != 'hold':
                self.logger.info(
                    f"⚫ HOLD {symbol}: Sinal fraco - confiança {signal['confidence']:.2f} < 0.7 necessário"
                )
            else:
                self.logger.info(f"⚫ HOLD {symbol}: Sem oportunidade de trade detectada")
                
        except Exception as e:
            self.logger.error(f"❌ Erro crítico na análise de {symbol}: {str(e)}")
            self.metrics['errors'] += 1
            self.metrics['last_error'] = str(e)

    def _execute_trade(self, symbol: str, signal: Dict, data):
        """
        Executa uma operação de trading com logs detalhados.
        """
        try:
            self.state = AgentState.TRADING
            
            # Obtém preço atual e calcula quantidade
            current_price = float(data['close'].iloc[-1])
            
            # Simula cálculo de quantidade baseado no risco (2% do portfólio por trade)
            portfolio_value = 10000  # Valor simulado do portfólio em USD
            risk_per_trade = 0.02    # 2% de risco por trade
            trade_value = portfolio_value * risk_per_trade
            quantity = trade_value / current_price
            
            action = signal['action'].upper()
            confidence = signal['confidence']
            
            self.logger.info(
                f"🎯 EXECUTANDO TRADE {symbol}: {action} | "
                f"Quantidade={quantity:.6f} | Preço=${current_price:.4f} | "
                f"Valor=${trade_value:.2f} | Confiança={confidence:.2f}"
            )
            
            # Simula execução no exchange
            result = self.strategy.execute_trade(symbol, signal, data)
            
            # Simula resultado baseado na confiança
            success = confidence > 0.8 and result  # Trade bem-sucedido se confiança alta
            
            if success:
                self.metrics['trades_executed'] += 1
                self.metrics['successful_trades'] += 1
                
                # Log de trade bem-sucedido
                self.log_trade_execution(
                    symbol, action, quantity, current_price, True,
                    f"Trade executado com sucesso - confiança {confidence:.2f}"
                )
                
                # Simula PnL (lucro/prejuízo)
                simulated_pnl = trade_value * (confidence - 0.7) * 2  # PnL baseado na confiança
                self.metrics['total_pnl'] += simulated_pnl
                
                self.logger.info(
                    f"💰 P&L SIMULADO {symbol}: ${simulated_pnl:+.2f} | "
                    f"P&L Total: ${self.metrics['total_pnl']:+.2f}"
                )
                
            else:
                self.metrics['trades_executed'] += 1
                
                # Log de trade falhado
                self.log_trade_execution(
                    symbol, action, quantity, current_price, False,
                    f"Trade falhado - problemas na execução ou baixa confiança"
                )
                
                self.logger.warning(f"⚠️ Trade {action} para {symbol} falhou na execução")
                
        except Exception as e:
            self.logger.error(f"❌ Erro crítico na execução de trade para {symbol}: {str(e)}")
            self.metrics['errors'] += 1

    def _calculate_portfolio_risk(self) -> float:
        """
        Calcula o risco atual do portfólio como porcentagem.
        """
        try:
            # Simula cálculo de risco baseado no número de trades ativos
            active_positions = len(self.positions)
            total_trades = self.metrics.get('trades_executed', 0)
            
            # Risco base de 2% por posição ativa + risco adicional por volume de trades
            base_risk = active_positions * 2.0  # 2% por posição
            volume_risk = min(total_trades * 0.1, 5.0)  # Máximo 5% adicional por volume
            
            total_risk = base_risk + volume_risk
            return min(total_risk, 25.0)  # Máximo 25% de risco
            
        except Exception:
            return 5.0  # Valor padrão conservador

    def _perform_demo_analysis(self, symbol: str):
        """
        Executa análise simulada para demonstração dos logs enriquecidos.
        """
        import random
        import time
        
        try:
            # Simula dados de mercado realistas
            base_prices = {
                'BTCUSDT': 45000 + random.uniform(-5000, 5000),
                'ETHUSDT': 3000 + random.uniform(-500, 500),
                'SOLUSDT': 100 + random.uniform(-20, 20)
            }
            
            current_price = base_prices.get(symbol, 1000 + random.uniform(-100, 100))
            price_change_24h = random.uniform(-8.0, 8.0)
            
            self.logger.info(f"📊 {symbol}: Preço=${current_price:.4f} | Variação 24h={price_change_24h:+.2f}% (dados simulados)")
            
            # Simula indicadores técnicos
            rsi = random.uniform(25, 75)
            macd = random.uniform(-0.5, 0.5)
            macd_signal = macd + random.uniform(-0.1, 0.1)
            ema_fast = current_price * (1 + random.uniform(-0.02, 0.02))
            ema_slow = current_price * (1 + random.uniform(-0.03, 0.03))
            
            indicators = {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow
            }
            
            # Analisa indicadores e gera fatores de decisão
            decision_factors = []
            
            if rsi > 70:
                decision_factors.append(f"RSI sobrecomprado ({rsi:.1f})")
            elif rsi < 30:
                decision_factors.append(f"RSI sobrevendido ({rsi:.1f})")
            else:
                decision_factors.append(f"RSI neutro ({rsi:.1f})")
            
            if macd > macd_signal:
                decision_factors.append("MACD bullish (cruzamento positivo)")
            else:
                decision_factors.append("MACD bearish (cruzamento negativo)")
            
            if ema_fast > ema_slow:
                decision_factors.append("Tendência de alta (EMA rápida > lenta)")
            else:
                decision_factors.append("Tendência de baixa (EMA rápida < lenta)")
            
            # Simula análise de volume
            volume_ratio = random.uniform(0.3, 2.5)
            if volume_ratio > 1.5:
                volume_analysis = f"Volume alto ({volume_ratio:.1f}x média)"
            elif volume_ratio < 0.5:
                volume_analysis = f"Volume baixo ({volume_ratio:.1f}x média)"
            else:
                volume_analysis = f"Volume normal ({volume_ratio:.1f}x média)"
            
            decision_factors.append(volume_analysis)
            
            # Determina decisão baseada nos indicadores
            bullish_signals = sum([
                rsi < 30,  # RSI sobrevendido
                macd > macd_signal,  # MACD positivo
                ema_fast > ema_slow,  # Tendência de alta
                price_change_24h > 2  # Alta nas últimas 24h
            ])
            
            bearish_signals = sum([
                rsi > 70,  # RSI sobrecomprado
                macd < macd_signal,  # MACD negativo
                ema_fast < ema_slow,  # Tendência de baixa
                price_change_24h < -2  # Queda nas últimas 24h
            ])
            
            if bullish_signals > bearish_signals:
                decision = "BUY"
                confidence = 0.6 + (bullish_signals * 0.1)
            elif bearish_signals > bullish_signals:
                decision = "SELL"
                confidence = 0.6 + (bearish_signals * 0.1)
            else:
                decision = "HOLD"
                confidence = 0.5
            
            # Log da estratégia detalhada
            self.log_strategy_decision(symbol, indicators, decision_factors, decision)
            
            # Log da análise de mercado completa
            analysis_data = {
                'indicators': indicators,
                'signal_strength': confidence,
                'volume_analysis': volume_analysis,
                'price_change_24h': price_change_24h,
                'reason': f"Análise demo baseada em: {', '.join(decision_factors)}"
            }
            self.log_market_analysis(symbol, current_price, analysis_data, decision)
            
            # Calcula risco do portfólio
            current_risk = self._calculate_portfolio_risk()
            max_risk = 10.0
            
            # Armazena análise
            self.last_analysis[symbol] = {
                'timestamp': datetime.now(),
                'signal': {'action': decision.lower(), 'confidence': confidence},
                'indicators': indicators,
                'price': current_price,
                'price_change_24h': price_change_24h,
                'decision_factors': decision_factors,
                'volume_analysis': volume_analysis
            }
            
            # Decisão de execução com gestão de risco
            if current_risk > max_risk:
                self.log_risk_management(
                    symbol, "BLOCK", current_risk, max_risk,
                    f"Bloqueando trades simulados - risco do portfólio muito alto"
                )
                return
            
            # Simula execução de trade se sinal for forte
            if decision != "HOLD" and confidence > 0.7:
                self.log_risk_management(
                    symbol, "APPROVE", current_risk, max_risk,
                    f"Trade simulado aprovado - risco dentro do limite (confiança: {confidence:.2f})"
                )
                self._execute_demo_trade(symbol, decision, current_price, confidence)
            elif decision != "HOLD":
                self.logger.info(
                    f"⚫ HOLD {symbol}: Sinal fraco simulado - confiança {confidence:.2f} < 0.7 necessário"
                )
            else:
                self.logger.info(f"⚫ HOLD {symbol}: Sem oportunidade de trade detectada (simulação)")
                
        except Exception as e:
            self.logger.error(f"❌ Erro na análise simulada de {symbol}: {str(e)}")
            self.metrics['errors'] += 1

    def _execute_demo_trade(self, symbol: str, action: str, price: float, confidence: float):
        """
        Simula execução de trade para demonstração.
        """
        try:
            self.state = AgentState.TRADING
            
            # Simula cálculo de quantidade
            portfolio_value = 10000  # Valor simulado
            risk_per_trade = 0.02    # 2% de risco
            trade_value = portfolio_value * risk_per_trade
            quantity = trade_value / price
            
            self.logger.info(
                f"🎯 TRADE SIMULADO {symbol}: {action} | "
                f"Quantidade={quantity:.6f} | Preço=${price:.4f} | "
                f"Valor=${trade_value:.2f} | Confiança={confidence:.2f}"
            )
            
            # Simula resultado
            success = confidence > 0.8
            
            if success:
                self.metrics['trades_executed'] += 1
                self.metrics['successful_trades'] += 1
                
                self.log_trade_execution(
                    symbol, action, quantity, price, True,
                    f"Trade simulado executado com sucesso - confiança {confidence:.2f}"
                )
                
                # Simula PnL
                simulated_pnl = trade_value * (confidence - 0.7) * 2
                self.metrics['total_pnl'] += simulated_pnl
                
                self.logger.info(
                    f"💰 P&L SIMULADO {symbol}: ${simulated_pnl:+.2f} | "
                    f"P&L Total: ${self.metrics['total_pnl']:+.2f}"
                )
            else:
                self.metrics['trades_executed'] += 1
                
                self.log_trade_execution(
                    symbol, action, quantity, price, False,
                    f"Trade simulado falhado - baixa confiança"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Erro na execução de trade simulado para {symbol}: {str(e)}")
            self.metrics['errors'] += 1

    def run_sync_loop(self):
        """
        Loop principal síncrono.
        """
        try:
            self.logger.info("Starting synchronous trading loop...")
            start_time = datetime.now()
            
            while self.running:
                try:
                    for symbol in TRADING_PAIRS:
                        self.analyze_and_trade(symbol)
                    
                    # Atualiza métricas
                    self.metrics['uptime'] = (datetime.now() - start_time).total_seconds()
                    
                    # Log de status periódico
                    if self.metrics['trades_executed'] % 10 == 0 and self.metrics['trades_executed'] > 0:
                        self.logger.info(f"Agent metrics: {self.metrics}")
                    
                    self.state = AgentState.ACTIVE
                    time.sleep(WORKER_INTERVAL)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    self.metrics['errors'] += 1
                    time.sleep(WORKER_INTERVAL)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in sync loop: {str(e)}")
            self.state = AgentState.ERROR
        finally:
            self.state = AgentState.STOPPING
            self.running = False

    async def run_async_loop(self):
        """
        Loop principal assíncrono com logs detalhados.
        """
        try:
            self.logger.info("🚀 Iniciando loop de trading assíncrono...")
            start_time = datetime.now()
            cycle_count = 0
            last_portfolio_log = datetime.now()
            
            # Inicializa fila de eventos
            self.event_queue = asyncio.Queue()
            
            # Log inicial das configurações
            self.logger.info(f"📋 CONFIGURAÇÃO INICIAL:")
            self.logger.info(f"   📊 Criptomoedas monitoradas: {', '.join(TRADING_PAIRS)}")
            self.logger.info(f"   ⏰ Intervalo de análise: {WORKER_INTERVAL}s")
            self.logger.info(f"   🎯 Timeframe: {TIMEFRAME}")
            self.logger.info(f"   🛡️ Risco máximo por trade: 2%")
            
            while self.running:
                try:
                    cycle_count += 1
                    cycle_start = datetime.now()
                    
                    self.logger.info(f"🔄 CICLO #{cycle_count} iniciado às {cycle_start.strftime('%H:%M:%S')}")
                    
                    # Processa cada símbolo
                    for i, symbol in enumerate(TRADING_PAIRS, 1):
                        self.logger.info(f"📈 [{i}/{len(TRADING_PAIRS)}] Analisando {symbol}...")
                        await asyncio.to_thread(self.analyze_and_trade, symbol)
                    
                    # Atualiza métricas
                    uptime_seconds = (datetime.now() - start_time).total_seconds()
                    self.metrics['uptime'] = uptime_seconds
                    
                    # Log de status do portfólio a cada 5 ciclos (aproximadamente 5 minutos se WORKER_INTERVAL=60)
                    if cycle_count % 5 == 0 or (datetime.now() - last_portfolio_log).total_seconds() > 300:
                        uptime_hours = uptime_seconds / 3600
                        success_rate = (self.metrics['successful_trades'] / max(self.metrics['trades_executed'], 1)) * 100
                        
                        # Simula dados do portfólio
                        portfolio_value = 10000 + self.metrics.get('total_pnl', 0)
                        current_risk = self._calculate_portfolio_risk()
                        
                        portfolio_positions = {
                            symbol: {
                                'size': 0.1 if symbol in self.last_analysis else 0,
                                'pnl': self.metrics.get('total_pnl', 0) / len(TRADING_PAIRS)
                            }
                            for symbol in TRADING_PAIRS
                        }
                        
                        self.log_portfolio_status(
                            portfolio_value, 
                            portfolio_positions, 
                            self.metrics.get('total_pnl', 0)
                        )
                        
                        # Log de métricas detalhadas
                        self.logger.info(
                            f"📊 MÉTRICAS CICLO #{cycle_count}: "
                            f"Uptime={uptime_hours:.1f}h | "
                            f"Trades={self.metrics['trades_executed']} | "
                            f"Taxa Sucesso={success_rate:.1f}% | "
                            f"Erros={self.metrics['errors']} | "
                            f"Risco={current_risk:.1f}%"
                        )
                        
                        last_portfolio_log = datetime.now()
                    
                    cycle_duration = (datetime.now() - cycle_start).total_seconds()
                    self.logger.info(f"✅ CICLO #{cycle_count} concluído em {cycle_duration:.1f}s")
                    
                    self.state = AgentState.ACTIVE
                    await asyncio.sleep(WORKER_INTERVAL)
                    
                except asyncio.CancelledError:
                    self.logger.info("🛑 Loop assíncrono cancelado")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Erro no loop assíncrono: {str(e)}")
                    self.metrics['errors'] += 1
                    await asyncio.sleep(WORKER_INTERVAL)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in async loop: {str(e)}")
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
        
        # Inicializa componentes
        if not self.initialize_components():
            self.logger.error("Failed to initialize agent")
            return
        
        self.running = True
        self.logger.info("Starting robust trading agent...")
        
        # Tenta modo assíncrono primeiro
        try:
            asyncio.run(self.run_async_loop())
        except Exception as e:
            self.logger.warning(f"Async mode failed: {e}, switching to sync mode")
            self.run_sync_loop()

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

# Aliases para compatibilidade
TradingWorker = RobustTradingAgent
IntelligentTradingAgent = RobustTradingAgent

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
    
    agent = RobustTradingAgent()
    agent.start()