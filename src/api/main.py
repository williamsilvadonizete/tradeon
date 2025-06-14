from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import secrets
import uvicorn
import logging
import sys
import json

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Configuração do logger (antes das importações)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.agent.worker import IntelligentTradingAgent
from src.agent.graph import MemoryGraph
from src.data.collector import DataCollector
from config.settings import API_KEY

# Import dos novos endpoints superinteligentes
try:
    import sys
    import os
    # Adiciona o diretório raiz ao path para importação
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.api.superintelligent_endpoints import router as superintelligent_router
    SUPERINTELLIGENT_AVAILABLE = True
    logger.info("✅ Superintelligent endpoints imported successfully")
except ImportError as e:
    logger.warning(f"❌ Superintelligent endpoints not available: {e}")
    SUPERINTELLIGENT_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Error importing superintelligent endpoints: {e}")
    SUPERINTELLIGENT_AVAILABLE = False

app = FastAPI(
    title="Superintelligent Crypto Trading Bot API",
    description="API para consulta de dados do sistema superinteligente de trading",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "trades",
            "description": "Operações relacionadas a trades",
        },
        {
            "name": "stats",
            "description": "Estatísticas e análises",
        },
        {
            "name": "superintelligent",
            "description": "Funcionalidades do sistema superinteligente",
        }
    ]
)

# Configuração de Segurança
API_KEY = os.getenv("API_KEY", secrets.token_hex(16))
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key_header

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclui routers dos endpoints superinteligentes
logger.info(f"🔍 SUPERINTELLIGENT_AVAILABLE = {SUPERINTELLIGENT_AVAILABLE}")
if SUPERINTELLIGENT_AVAILABLE:
    try:
        app.include_router(superintelligent_router, dependencies=[Depends(get_api_key)])
        logger.info("✅ Superintelligent router included successfully")
        logger.info(f"🔗 Router routes: {[route.path for route in superintelligent_router.routes]}")
    except Exception as e:
        logger.error(f"❌ Error including superintelligent router: {e}")
        SUPERINTELLIGENT_AVAILABLE = False
else:
    logger.warning("❌ Superintelligent endpoints NOT loaded")

# Conexão com o PostgreSQL
def get_db_connection():
    try:
        # Suporte para ambos os formatos: DATABASE_URL e variáveis separadas
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Formato DATABASE_URL (desenvolvimento local)
            conn = psycopg2.connect(database_url)
        else:
            # Variáveis separadas (produção ECS com Secrets)
            conn = psycopg2.connect(
                dbname=os.getenv("DATABASE_NAME", "cryptotrader"),
                user=os.getenv("DATABASE_USER", "cryptotrader"),
                password=os.getenv("DATABASE_PASSWORD"),
                host=os.getenv("DATABASE_HOST", "localhost"),
                port=os.getenv("DATABASE_PORT", "5432")
            )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# Modelos Pydantic
class Trade(BaseModel):
    id: Optional[int]
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl: Optional[float]
    confidence: float
    reason: str
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "action": "buy",
                "entry_price": 50000.0,
                "exit_price": 51000.0,
                "position_size": 0.1,
                "pnl": 100.0,
                "confidence": 0.85,
                "reason": "Sinal técnico forte de compra"
            }
        }

class TradeStats(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    best_trade: float
    worst_trade: float

    class Config:
        json_schema_extra = {
            "example": {
                "total_trades": 100,
                "winning_trades": 60,
                "losing_trades": 40,
                "win_rate": 0.6,
                "total_pnl": 1500.50,
                "average_pnl": 15.00,
                "best_trade": 200.00,
                "worst_trade": -50.00
            }
        }

# Endpoints
@app.get("/", tags=["root"])
async def root():
    """
    Endpoint raiz da API.
    Retorna uma mensagem de boas-vindas.
    """
    return {"message": "Crypto Trading Bot API"}

@app.get("/health", tags=["root"])
async def health():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "message": "API is running"}

@app.post("/migrate", tags=["admin"], dependencies=[Depends(get_api_key)])
async def run_migrations():
    """
    Executa as migrações do banco de dados.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Cria as tabelas
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL,
            entry_price DECIMAL(20, 8) NOT NULL,
            exit_price DECIMAL(20, 8),
            position_size DECIMAL(20, 8) NOT NULL,
            pnl DECIMAL(20, 8),
            confidence DECIMAL(5, 2) NOT NULL,
            reason TEXT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            total_trades INTEGER NOT NULL,
            winning_trades INTEGER NOT NULL,
            losing_trades INTEGER NOT NULL,
            win_rate DECIMAL(5, 2) NOT NULL,
            total_pnl DECIMAL(20, 8) NOT NULL,
            average_pnl DECIMAL(20, 8) NOT NULL,
            best_trade DECIMAL(20, 8) NOT NULL,
            worst_trade DECIMAL(20, 8) NOT NULL,
            period_start TIMESTAMP NOT NULL,
            period_end TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS configurations (
            id SERIAL PRIMARY KEY,
            key VARCHAR(50) NOT NULL UNIQUE,
            value JSONB NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            level VARCHAR(10) NOT NULL,
            message TEXT NOT NULL,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Cria índices
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_symbol ON performance(symbol);
        CREATE INDEX IF NOT EXISTS idx_performance_period ON performance(period_start, period_end);
        CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
        CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at);
        """)
        
        # Cria função para atualizar updated_at
        cur.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """)
        
        # Cria triggers
        cur.execute("""
        DROP TRIGGER IF EXISTS update_trades_updated_at ON trades;
        CREATE TRIGGER update_trades_updated_at
            BEFORE UPDATE ON trades
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
            
        DROP TRIGGER IF EXISTS update_configurations_updated_at ON configurations;
        CREATE TRIGGER update_configurations_updated_at
            BEFORE UPDATE ON configurations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)
        
        # Insere configurações iniciais
        cur.execute("""
        INSERT INTO configurations (key, value, description)
        VALUES 
            ('trading_pairs', '["BTCUSDT", "ETHUSDT"]', 'Lista de pares de trading'),
            ('timeframe', '"1h"', 'Timeframe padrão'),
            ('stop_loss', '{"strategy": "atr", "atr_period": 14, "atr_multiplier": 2.0}', 'Configurações de stop loss'),
            ('risk_management', '{"max_position_size": 0.1, "max_drawdown": 0.05}', 'Configurações de gestão de risco')
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value,
            updated_at = CURRENT_TIMESTAMP
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("Database migration completed successfully")
        return {"status": "success", "message": "Database migration completed successfully"}
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

@app.get("/trades", response_model=List[Trade], tags=["trades"], dependencies=[Depends(get_api_key)])
async def get_trades(
    symbol: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Retorna a lista de trades com filtros opcionais.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    query = "SELECT * FROM trades WHERE 1=1"
    params = []
    
    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)
    if action:
        query += " AND action = %s"
        params.append(action)
        
    query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    try:
        cur.execute(query, params)
        result = cur.fetchall()
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Error fetching trades: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.get("/trades/stats", response_model=TradeStats, tags=["stats"], dependencies=[Depends(get_api_key)])
async def get_trade_stats(symbol: Optional[str] = None):
    """
    Retorna estatísticas gerais dos trades.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT 
        COUNT(*) as total_trades,
        COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) as winning_trades,
        COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losing_trades,
        COALESCE(AVG(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0.0) as win_rate,
        COALESCE(SUM(pnl), 0.0) as total_pnl,
        COALESCE(AVG(pnl), 0.0) as average_pnl,
        COALESCE(MAX(pnl), 0.0) as best_trade,
        COALESCE(MIN(pnl), 0.0) as worst_trade
    FROM trades
    WHERE 1=1
    """
    params = []
    
    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)
    
    try:
        cur.execute(query, params)
        result = cur.fetchone()
        return dict(result)
    except Exception as e:
        logger.error(f"Error fetching trade stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.get("/trades/symbols", tags=["trades"], dependencies=[Depends(get_api_key)])
async def get_symbols():
    """
    Retorna lista de símbolos únicos disponíveis.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT DISTINCT symbol FROM trades ORDER BY symbol")
        result = cur.fetchall()
        return [row[0] for row in result]
    except Exception as e:
        logger.error(f"Error fetching symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.get("/trades/daily-stats", tags=["stats"], dependencies=[Depends(get_api_key)])
async def get_daily_stats(symbol: Optional[str] = None):
    """
    Retorna estatísticas diárias dos trades.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT 
        DATE(timestamp) as date,
        COUNT(*) as total_trades,
        SUM(pnl) as daily_pnl,
        AVG(pnl) as avg_pnl,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades
    FROM trades
    WHERE 1=1
    """
    params = []
    
    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)
    
    query += " GROUP BY DATE(timestamp) ORDER BY date DESC"
    
    try:
        cur.execute(query, params)
        result = cur.fetchall()
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Error fetching daily stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# Inicializa componentes globais (conexões de banco sob demanda)
agent = None
memory_graph = None

def get_agent():
    global agent
    if agent is None:
        agent = IntelligentTradingAgent()
    return agent

def get_memory_graph():
    global memory_graph
    if memory_graph is None:
        memory_graph = MemoryGraph()
    return memory_graph

@app.post("/agent/start", dependencies=[Depends(get_api_key)])
async def start_agent():
    """
    Inicia o agente de trading.
    """
    try:
        # Inicia o agente em background
        import asyncio
        trading_agent = get_agent()
        asyncio.create_task(trading_agent.run_agent())
        return {"status": "success", "message": "Trading agent started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/stop", dependencies=[Depends(get_api_key)])
async def stop_agent():
    """
    Para o agente de trading.
    """
    try:
        trading_agent = get_agent()
        trading_agent.stop()
        return {"status": "success", "message": "Trading agent stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status", dependencies=[Depends(get_api_key)])
async def agent_status():
    """
    Retorna o status atual do agente.
    """
    trading_agent = get_agent()
    return trading_agent.get_status()

# Novos endpoints para o app mobile
@app.get("/portfolio", dependencies=[Depends(get_api_key)])
async def get_portfolio():
    """
    Retorna informações do portfólio para o app mobile.
    """
    try:
        # Simulação de dados do portfólio
        portfolio = {
            "total_balance": 10000.00,
            "available_balance": 8500.00,
            "pnl_24h": 150.25,
            "pnl_percentage_24h": 1.53,
            "assets": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "amount": 0.15,
                    "value_usd": 6750.00,
                    "price_usd": 45000.00,
                    "change_24h": 2.5
                },
                {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "amount": 2.8,
                    "value_usd": 5600.00,
                    "price_usd": 2000.00,
                    "change_24h": -1.2
                },
                {
                    "symbol": "SOL",
                    "name": "Solana",
                    "amount": 50.0,
                    "value_usd": 2500.00,
                    "price_usd": 50.00,
                    "change_24h": 5.8
                }
            ]
        }
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market", dependencies=[Depends(get_api_key)])
async def get_market_data():
    """
    Retorna dados do mercado para o app mobile.
    """
    try:
        from src.exchanges.bybit import BybitAdapter
        from config.settings import EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET
        
        exchange = BybitAdapter(EXCHANGE_API_KEY, EXCHANGE_SECRET, TESTNET)
        collector = DataCollector(exchange)
        
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        market_data = await collector.get_market_overview(symbols)
        
        return {"market_data": market_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/manual", dependencies=[Depends(get_api_key)])
async def execute_manual_trade(
    symbol: str,
    action: str,
    amount: float,
    order_type: str = "market"
):
    """
    Executa um trade manual via app mobile.
    """
    try:
        # Validações básicas
        if action not in ["buy", "sell"]:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")
        
        # Simula execução do trade
        trade_result = {
            "id": f"trade_{datetime.now().timestamp()}",
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "order_type": order_type,
            "status": "executed",
            "timestamp": datetime.now().isoformat(),
            "price": 45000.00 if symbol == "BTCUSDT" else 2000.00  # Preços simulados
        }
        
        return {"trade": trade_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions", dependencies=[Depends(get_api_key)])
async def get_transactions(
    limit: int = 50,
    offset: int = 0,
    symbol: Optional[str] = None
):
    """
    Retorna histórico de transações para o app mobile.
    """
    try:
        # Simula dados de transações
        transactions = [
            {
                "id": f"tx_{i}",
                "type": "trade" if i % 2 == 0 else "deposit",
                "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"][i % 3],
                "amount": 0.1 + (i * 0.01),
                "price": 45000 - (i * 100),
                "value_usd": (0.1 + (i * 0.01)) * (45000 - (i * 100)),
                "fee": 2.5,
                "status": "completed",
                "timestamp": (datetime.now().timestamp() - (i * 3600))
            }
            for i in range(limit)
        ]
        
        if symbol:
            transactions = [tx for tx in transactions if tx["symbol"] == symbol]
        
        return {
            "transactions": transactions[offset:offset+limit],
            "total": len(transactions),
            "has_more": offset + limit < len(transactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading-stats", dependencies=[Depends(get_api_key)])
async def get_trading_statistics():
    """
    Retorna estatísticas de trading para o app mobile.
    """
    try:
        stats = {
            "performance": {
                "total_trades": 156,
                "winning_trades": 98,
                "losing_trades": 58,
                "win_rate": 62.8,
                "profit_factor": 1.45,
                "sharpe_ratio": 1.23
            },
            "pnl": {
                "total_pnl": 2847.65,
                "realized_pnl": 2500.00,
                "unrealized_pnl": 347.65,
                "today_pnl": 125.30,
                "week_pnl": 456.78,
                "month_pnl": 1203.45
            },
            "risk_metrics": {
                "max_drawdown": 0.0,
                "current_drawdown": -45.23,
                "var_95": -89.34,
                "expected_shortfall": -156.78
            },
            "trading_activity": {
                "trades_today": 3,
                "trades_week": 18,
                "trades_month": 67,
                "avg_trade_size": 0.0,
                "avg_holding_time": "2h 15m"
            }
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/insights", dependencies=[Depends(get_api_key)])
async def get_agent_insights():
    """
    Retorna insights do agente inteligente para o app mobile.
    """
    try:
        insights = {
            "market_sentiment": {
                "overall": "bullish",
                "confidence": 0.78,
                "factors": [
                    "Strong technical indicators",
                    "Positive momentum",
                    "Volume increasing"
                ]
            },
            "recommendations": [
                {
                    "symbol": "BTCUSDT",
                    "action": "buy",
                    "confidence": 0.85,
                    "reasoning": "RSI oversold, MACD bullish crossover",
                    "target_price": 47000,
                    "stop_loss": 43000
                },
                {
                    "symbol": "ETHUSDT",
                    "action": "hold",
                    "confidence": 0.65,
                    "reasoning": "Consolidation phase, waiting for breakout",
                    "target_price": 2200,
                    "stop_loss": 1900
                }
            ],
            "risk_assessment": {
                "market_volatility": "medium",
                "correlation_risk": "low",
                "liquidity_risk": "low",
                "overall_risk": "medium"
            },
            "learning_progress": {
                "patterns_learned": 47,
                "accuracy_improvement": 12.5,
                "confidence_trend": "increasing",
                "last_update": datetime.now().isoformat()
            }
        }
        return insights
    except Exception as e:
        logger.error(f"Error getting agent insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/settings", dependencies=[Depends(get_api_key)])
async def get_app_settings():
    """
    Retorna configurações do app para o mobile.
    """
    try:
        settings = {
            "trading": {
                "auto_trading_enabled": get_agent().is_running() if agent else False,
                "risk_level": "medium",
                "max_position_size": 0.1,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 4.0
            },
            "notifications": {
                "trade_alerts": True,
                "price_alerts": True,
                "system_alerts": True,
                "daily_summary": True
            },
            "display": {
                "theme": "dark",
                "currency": "USD",
                "precision": 6,
                "chart_interval": "1h"
            },
            "security": {
                "biometric_enabled": False,
                "session_timeout": 30,
                "two_factor_enabled": False
            }
        }
        return settings
    except Exception as e:
        logger.error(f"Error getting app settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/settings", dependencies=[Depends(get_api_key)])
async def update_app_settings(settings: Dict[str, Any]):
    """
    Atualiza configurações do app.
    """
    try:
        # Aqui você salvaria as configurações no banco ou arquivo
        logger.info(f"Settings updated: {settings}")
        return {"message": "Settings updated successfully", "settings": settings}
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_startup_migrations():
    """Run database migrations on startup."""
    try:
        logger.info("Running database migrations on startup...")
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Cria as tabelas
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL,
            entry_price DECIMAL(20, 8) NOT NULL,
            exit_price DECIMAL(20, 8),
            position_size DECIMAL(20, 8) NOT NULL,
            pnl DECIMAL(20, 8),
            confidence DECIMAL(5, 2) NOT NULL,
            reason TEXT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            total_trades INTEGER NOT NULL,
            winning_trades INTEGER NOT NULL,
            losing_trades INTEGER NOT NULL,
            win_rate DECIMAL(5, 2) NOT NULL,
            total_pnl DECIMAL(20, 8) NOT NULL,
            average_pnl DECIMAL(20, 8) NOT NULL,
            best_trade DECIMAL(20, 8) NOT NULL,
            worst_trade DECIMAL(20, 8) NOT NULL,
            period_start TIMESTAMP NOT NULL,
            period_end TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS configurations (
            id SERIAL PRIMARY KEY,
            key VARCHAR(50) NOT NULL UNIQUE,
            value JSONB NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            level VARCHAR(10) NOT NULL,
            message TEXT NOT NULL,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Cria índices
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_symbol ON performance(symbol);
        CREATE INDEX IF NOT EXISTS idx_performance_period ON performance(period_start, period_end);
        CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
        CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at);
        """)
        
        # Cria função para atualizar updated_at
        cur.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """)
        
        # Cria triggers
        cur.execute("""
        DROP TRIGGER IF EXISTS update_trades_updated_at ON trades;
        CREATE TRIGGER update_trades_updated_at
            BEFORE UPDATE ON trades
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
            
        DROP TRIGGER IF EXISTS update_configurations_updated_at ON configurations;
        CREATE TRIGGER update_configurations_updated_at
            BEFORE UPDATE ON configurations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)
        
        # Insere configurações iniciais
        cur.execute("""
        INSERT INTO configurations (key, value, description)
        VALUES 
            ('trading_pairs', '["BTCUSDT", "ETHUSDT"]', 'Lista de pares de trading'),
            ('timeframe', '"1h"', 'Timeframe padrão'),
            ('stop_loss', '{"strategy": "atr", "atr_period": 14, "atr_multiplier": 2.0}', 'Configurações de stop loss'),
            ('risk_management', '{"max_position_size": 0.1, "max_drawdown": 0.05}', 'Configurações de gestão de risco')
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value,
            updated_at = CURRENT_TIMESTAMP
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("Database migrations completed successfully on startup")
        
    except Exception as e:
        logger.error(f"Startup migration failed: {str(e)}")
        # Don't fail startup if migrations fail - tables might already exist

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    await run_startup_migrations()

if __name__ == "__main__":
    print(f"API Key: {API_KEY}")  # Mostra a chave API no console ao iniciar
    uvicorn.run(app, host="0.0.0.0", port=8000) 