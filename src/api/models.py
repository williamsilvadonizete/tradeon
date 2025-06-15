from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Numeric, Enum, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

class TradeStatus(str, enum.Enum):
    """Status possíveis para um trade."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class TradeSide(str, enum.Enum):
    """Lado do trade (compra/venda)."""
    BUY = "buy"
    SELL = "sell"

class Trade(Base):
    """Modelo para trades."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange_id = Column(String(50), nullable=False, index=True)
    side = Column(Enum(TradeSide), nullable=False)
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.PENDING)
    entry_price = Column(Numeric(20, 8), nullable=False)
    exit_price = Column(Numeric(20, 8))
    position_size = Column(Numeric(20, 8), nullable=False)
    pnl = Column(Numeric(20, 8))
    confidence = Column(Float, nullable=False)
    reason = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class WorkerLog(Base):
    """Modelo para logs do worker."""
    __tablename__ = "worker_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange_id = Column(String(50), nullable=False, index=True)
    level = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TradeStats(Base):
    """Modelo para estatísticas de trading."""
    __tablename__ = "trade_stats"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange_id = Column(String(50), nullable=False, index=True)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    total_profit = Column(Numeric(20, 8), nullable=False, default=0)
    win_rate = Column(Numeric(20, 8), nullable=False, default=0)
    average_profit = Column(Numeric(20, 8), nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Modelos Pydantic para validação de entrada/saída
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class TradeBase(BaseModel):
    """Modelo base para trades."""
    symbol: str
    exchange_id: str
    side: TradeSide
    entry_price: float
    position_size: float
    confidence: float
    reason: Optional[str] = None

class TradeCreate(TradeBase):
    """Modelo para criação de trades."""
    pass

class TradeResponse(TradeBase):
    """Modelo para resposta de trades."""
    id: int
    status: TradeStatus
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class WorkerLogBase(BaseModel):
    """Modelo base para logs."""
    symbol: str
    exchange_id: str
    level: str
    message: str
    details: Optional[str] = None

class WorkerLogCreate(WorkerLogBase):
    """Modelo para criação de logs."""
    pass

class WorkerLogResponse(WorkerLogBase):
    """Modelo para resposta de logs."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TradeStatsBase(BaseModel):
    """Modelo base para estatísticas."""
    symbol: str
    exchange_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    win_rate: float
    average_profit: float

class TradeStatsCreate(TradeStatsBase):
    """Modelo para criação de estatísticas."""
    pass

class TradeStatsResponse(TradeStatsBase):
    """Modelo para resposta de estatísticas."""
    id: int
    created_at: datetime
    last_updated: datetime

    class Config:
        from_attributes = True

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    timeframe = Column(String)
    indicators = Column(JSON, nullable=True)

class Portfolio(Base):
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    amount = Column(Float)
    average_entry = Column(Float)
    current_price = Column(Float)
    pnl = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    status = Column(String)  # active, closed

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    type = Column(String)  # deposit, withdrawal, trade
    amount = Column(Float)
    price = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String)  # completed, pending, failed
    details = Column(JSON, nullable=True)

class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(JSON)
    description = Column(String, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

class AgentInsight(Base):
    __tablename__ = "agent_insights"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    insight_type = Column(String)  # market, technical, sentiment
    content = Column(JSON)
    confidence = Column(Float)
    source = Column(String)
    relevance = Column(Float)
    status = Column(String)  # active, archived 