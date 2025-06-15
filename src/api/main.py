from fastapi import FastAPI, HTTPException, Security, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional, Dict, Any
import os
from datetime import datetime, timedelta
import secrets
import uvicorn
import logging
import sys
from src.api.models import (
    Trade, WorkerLog, TradeStats,
    TradeCreate, TradeResponse,
    WorkerLogResponse, TradeStatsResponse
)
from src.api.database import get_db, init_db
from config.settings import API_KEY

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Trading Bot API",
    description="API para consulta de dados do sistema de trading",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "trades",
            "description": "Operações relacionadas a trades",
        },
        {
            "name": "logs",
            "description": "Logs e análises do worker",
        },
        {
            "name": "stats",
            "description": "Estatísticas de trading",
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

# Endpoints
@app.get("/", tags=["root"])
async def root():
    """Endpoint raiz da API."""
    return {"message": "Crypto Trading Bot API"}

@app.get("/health", tags=["root"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}

# Endpoints de Trades
@app.get("/trades", response_model=List[TradeResponse], tags=["trades"])
async def get_trades(
    db: AsyncSession = Depends(get_db),
    symbol: Optional[str] = None,
    exchange_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key: str = Depends(get_api_key)
):
    """Lista todos os trades com filtros opcionais."""
    try:
        query = select(Trade)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        if exchange_id:
            query = query.filter(Trade.exchange_id == exchange_id)
        if status:
            query = query.filter(Trade.status == status)
            
        query = query.order_by(Trade.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error in get_trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/trades/{trade_id}", response_model=TradeResponse, tags=["trades"])
async def get_trade(
    trade_id: int,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Obtém detalhes de um trade específico."""
    try:
        query = select(Trade).filter(Trade.id == trade_id)
        result = await db.execute(query)
        trade = result.scalar_one_or_none()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        return trade
    except Exception as e:
        logger.error(f"Error in get_trade: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Endpoints de Logs
@app.get("/logs", response_model=List[WorkerLogResponse], tags=["logs"])
async def get_logs(
    db: AsyncSession = Depends(get_db),
    symbol: Optional[str] = None,
    exchange_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key: str = Depends(get_api_key)
):
    """Lista os logs do worker com filtros opcionais."""
    try:
        query = select(WorkerLog)
        
        if symbol:
            query = query.filter(WorkerLog.symbol == symbol)
        if exchange_id:
            query = query.filter(WorkerLog.exchange_id == exchange_id)
            
        query = query.order_by(WorkerLog.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error in get_logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/logs/{log_id}", response_model=WorkerLogResponse, tags=["logs"])
async def get_log(
    log_id: int,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Obtém detalhes de um log específico."""
    try:
        query = select(WorkerLog).filter(WorkerLog.id == log_id)
        result = await db.execute(query)
        log = result.scalar_one_or_none()
        if not log:
            raise HTTPException(status_code=404, detail="Log not found")
        return log
    except Exception as e:
        logger.error(f"Error in get_log: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Endpoints de Estatísticas
@app.get("/trades/stats", response_model=List[TradeStatsResponse], tags=["stats"])
async def get_stats(
    db: AsyncSession = Depends(get_db),
    symbol: Optional[str] = None,
    exchange_id: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """Get trading statistics."""
    try:
        logger.info("Starting stats search...")
        # Construir a query base
        query = select(TradeStats)
        logger.info("Base query created")
        
        # Aplicar filtros se fornecidos
        if symbol:
            query = query.where(TradeStats.symbol == symbol)
            logger.info(f"Added symbol filter: {symbol}")
        if exchange_id:
            query = query.where(TradeStats.exchange_id == exchange_id)
            logger.info(f"Added exchange_id filter: {exchange_id}")
        
        # Executar a query
        logger.info("Executing query...")
        try:
            result = await db.execute(query)
            stats = result.scalars().all()
            logger.info(f"Query executed successfully. Found {len(stats)} results")
            return stats
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats/{symbol}", response_model=TradeStatsResponse, tags=["stats"])
async def get_symbol_stats(
    symbol: str,
    exchange_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Get statistics for a specific symbol."""
    try:
        query = select(TradeStats).filter(
            TradeStats.symbol == symbol,
            TradeStats.exchange_id == exchange_id
        )
        result = await db.execute(query)
        stats = result.scalar_one_or_none()
        if not stats:
            raise HTTPException(status_code=404, detail="Statistics not found")
        return stats
    except Exception as e:
        logger.error(f"Error in get_symbol_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Inicializa o banco de dados na inicialização da aplicação."""
    try:
        await init_db()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 