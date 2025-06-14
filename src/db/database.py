"""
Database connection and operations for real trading data
"""
import os
import asyncio
import asyncpg
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pool variable
_pool: Optional[asyncpg.Pool] = None

async def get_db_connection() -> Optional[asyncpg.Pool]:
    """Get database connection pool."""
    global _pool
    
    if _pool is None:
        try:
            # Log connection parameters (without sensitive data)
            logger.info("Attempting to connect to database with parameters:")
            logger.info(f"Host: {os.getenv('POSTGRES_HOST', 'db')}")
            logger.info(f"Port: {os.getenv('POSTGRES_PORT', '5432')}")
            logger.info(f"Database: {os.getenv('POSTGRES_DB', 'trading_bot')}")
            logger.info(f"User: {os.getenv('POSTGRES_USER', 'postgres')}")
            
            _pool = await asyncpg.create_pool(
                host=os.getenv('POSTGRES_HOST', 'db'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
                database=os.getenv('POSTGRES_DB', 'trading_bot'),
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            
            # Test connection
            async with _pool.acquire() as conn:
                version = await conn.fetchval('SELECT version()')
                logger.info(f"✅ Database connection pool created successfully")
                logger.info(f"📊 PostgreSQL version: {version}")
                
        except Exception as e:
            logger.error(f"❌ Error creating database connection pool: {str(e)}")
            logger.error(f"Connection parameters:")
            logger.error(f"  Host: {os.getenv('POSTGRES_HOST', 'db')}")
            logger.error(f"  Port: {os.getenv('POSTGRES_PORT', '5432')}")
            logger.error(f"  Database: {os.getenv('POSTGRES_DB', 'trading_bot')}")
            logger.error(f"  User: {os.getenv('POSTGRES_USER', 'postgres')}")
            return None
            
    return _pool

async def close_db_connection():
    """Close database connection pool."""
    global _pool
    
    if _pool:
        try:
            await _pool.close()
            _pool = None
            logger.info("✅ Database connection pool closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing database connection pool: {str(e)}")

async def execute_query(query: str, *args) -> None:
    """Execute a query."""
    if not _pool:
        logger.error("❌ Database connection pool not initialized")
        return
        
    try:
        async with _pool.acquire() as conn:
            await conn.execute(query, *args)
    except Exception as e:
        logger.error(f"❌ Error executing query: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Args: {args}")
        raise

async def fetch_one(query: str, *args) -> Optional[Dict[str, Any]]:
    """Fetch one row from database."""
    if not _pool:
        logger.error("❌ Database connection pool not initialized")
        return None
        
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"❌ Error fetching one row: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Args: {args}")
        raise

async def fetch_all(query: str, *args) -> List[Dict[str, Any]]:
    """Fetch all rows from database."""
    if not _pool:
        logger.error("❌ Database connection pool not initialized")
        return []
        
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"❌ Error fetching all rows: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Args: {args}")
        raise

class Database:
    def __init__(self):
        self.pool = None
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "trading_bot")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
    async def connect(self):
        """Connect to PostgreSQL database"""
        try:
            logger.info("Attempting to connect to database with parameters:")
            logger.info(f"Host: {self.host}")
            logger.info(f"Port: {self.port}")
            logger.info(f"Database: {self.database}")
            logger.info(f"User: {self.user}")
            
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=5,
                command_timeout=10
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval('SELECT version()')
                logger.info(f"✅ Connected to PostgreSQL database: {self.database}")
                logger.info(f"📊 PostgreSQL version: {version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            logger.error(f"Connection parameters:")
            logger.error(f"  Host: {self.host}")
            logger.error(f"  Port: {self.port}")
            logger.error(f"  Database: {self.database}")
            logger.error(f"  User: {self.user}")
            return False
            
    async def disconnect(self):
        """Disconnect from database"""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict]:
        """Fetch one record from database"""
        if not self.pool:
            logger.warning("Database not connected")
            return None
            
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(query, *args)
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Database fetch_one error: {e}")
            return None
    
    async def fetch_all(self, query: str, *args) -> List[Dict]:
        """Fetch all records from database"""
        if not self.pool:
            logger.warning("Database not connected")
            return []
            
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch(query, *args)
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Database fetch_all error: {e}")
            return []
    
    async def execute(self, query: str, *args) -> bool:
        """Execute query (INSERT, UPDATE, DELETE)"""
        if not self.pool:
            logger.warning("Database not connected")
            return False
            
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, *args)
                return True
        except Exception as e:
            logger.error(f"Database execute error: {e}")
            return False
    
    async def insert_trade(self, trade_data: Dict) -> bool:
        """Insert new trade record"""
        query = """
        INSERT INTO trades (symbol, action, entry_price, exit_price, position_size, 
                           pnl, confidence, reason, timestamp, status)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        return await self.execute(
            query,
            trade_data.get('symbol'),
            trade_data.get('action'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('position_size'),
            trade_data.get('pnl'),
            trade_data.get('confidence'),
            trade_data.get('reason'),
            trade_data.get('timestamp', datetime.now(timezone.utc)),
            trade_data.get('status', 'completed')
        )
    
    async def update_performance_metrics(self, metrics: Dict) -> bool:
        """Update performance metrics"""
        query = """
        INSERT INTO performance (win_rate, total_pnl, average_pnl, best_trade, worst_trade, 
                               total_trades, best_strategy, timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id) DO UPDATE SET
            win_rate = EXCLUDED.win_rate,
            total_pnl = EXCLUDED.total_pnl,
            average_pnl = EXCLUDED.average_pnl,
            best_trade = EXCLUDED.best_trade,
            worst_trade = EXCLUDED.worst_trade,
            total_trades = EXCLUDED.total_trades,
            best_strategy = EXCLUDED.best_strategy,
            timestamp = EXCLUDED.timestamp
        """
        
        return await self.execute(
            query,
            metrics.get('win_rate'),
            metrics.get('total_pnl'),
            metrics.get('average_pnl'),
            metrics.get('best_trade'),
            metrics.get('worst_trade'),
            metrics.get('total_trades'),
            metrics.get('best_strategy'),
            datetime.now(timezone.utc)
        )
    
    async def get_today_trades(self) -> List[Dict]:
        """Get today's trades"""
        query = """
        SELECT * FROM trades 
        WHERE DATE(timestamp) = CURRENT_DATE 
        ORDER BY timestamp DESC
        """
        return await self.fetch_all(query)
    
    async def get_latest_performance(self) -> Optional[Dict]:
        """Get latest performance metrics"""
        query = """
        SELECT * FROM performance 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        return await self.fetch_one(query)
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            result = await self.fetch_one("SELECT 1 as status")
            return result is not None
        except:
            return False