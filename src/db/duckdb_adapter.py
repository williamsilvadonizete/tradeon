"""
DuckDB adapter for real trading data
"""
import duckdb
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DuckDBAdapter:
    def __init__(self):
        self.db_path = Path(__file__).parent.parent.parent / "data" / "trades.duckdb"
        self.conn = None
        
    async def connect(self):
        """Connect to DuckDB database"""
        try:
            self.conn = duckdb.connect(str(self.db_path))
            logger.info(f"✅ Connected to DuckDB database: {self.db_path}")
            
            # Ensure tables exist
            await self.create_tables()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to DuckDB: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")
    
    async def create_tables(self):
        """Create tables if they don't exist"""
        try:
            # Trades table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR,
                action VARCHAR,
                entry_price DOUBLE,
                exit_price DOUBLE,
                position_size DOUBLE,
                pnl DOUBLE,
                confidence DOUBLE,
                reason VARCHAR,
                timestamp TIMESTAMP,
                status VARCHAR DEFAULT 'completed'
            )
            """)
            
            # Performance table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY,
                win_rate DOUBLE,
                total_pnl DOUBLE,
                average_pnl DOUBLE,
                best_trade DOUBLE,
                worst_trade DOUBLE,
                total_trades INTEGER,
                best_strategy VARCHAR,
                timestamp TIMESTAMP
            )
            """)
            
            # AI insights table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY,
                sentiment VARCHAR,
                confidence DOUBLE,
                patterns_active INTEGER,
                learning_status VARCHAR,
                accuracy DOUBLE,
                timestamp TIMESTAMP
            )
            """)
            
            # Risk metrics table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY,
                portfolio_heat DOUBLE,
                max_drawdown DOUBLE,
                var_95 DOUBLE,
                correlation_risk VARCHAR,
                timestamp TIMESTAMP
            )
            """)
            
            logger.info("✅ Database tables verified/created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict]:
        """Fetch one record from database"""
        if not self.conn:
            logger.warning("Database not connected")
            return None
            
        try:
            result = self.conn.execute(query).fetchone()
            if result:
                # Get column names
                columns = [desc[0] for desc in self.conn.description]
                return dict(zip(columns, result))
            return None
        except Exception as e:
            logger.error(f"DuckDB fetch_one error: {e}")
            return None
    
    async def fetch_all(self, query: str, *args) -> List[Dict]:
        """Fetch all records from database"""
        if not self.conn:
            logger.warning("Database not connected")
            return []
            
        try:
            results = self.conn.execute(query).fetchall()
            if results:
                # Get column names
                columns = [desc[0] for desc in self.conn.description]
                return [dict(zip(columns, row)) for row in results]
            return []
        except Exception as e:
            logger.error(f"DuckDB fetch_all error: {e}")
            return []
    
    async def execute(self, query: str, *args) -> bool:
        """Execute query (INSERT, UPDATE, DELETE)"""
        if not self.conn:
            logger.warning("Database not connected")
            return False
            
        try:
            self.conn.execute(query)
            return True
        except Exception as e:
            logger.error(f"DuckDB execute error: {e}")
            return False
    
    async def insert_trade(self, trade_data: Dict) -> bool:
        """Insert new trade record"""
        query = """
        INSERT INTO trades (symbol, action, entry_price, exit_price, position_size, 
                           pnl, confidence, reason, timestamp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            self.conn.execute(query, (
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
            ))
            return True
        except Exception as e:
            logger.error(f"Failed to insert trade: {e}")
            return False
    
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
    
    async def insert_sample_data(self):
        """Insert sample trades for testing"""
        try:
            # Insert some sample trades for today
            sample_trades = [
                {
                    'symbol': 'BTCUSDT',
                    'action': 'BUY',
                    'entry_price': 67500.0,
                    'exit_price': 68200.0,
                    'position_size': 0.01,
                    'pnl': 7.0,
                    'confidence': 0.85,
                    'reason': 'RSI oversold + MACD bullish divergence',
                    'timestamp': datetime.now(timezone.utc),
                    'status': 'completed'
                },
                {
                    'symbol': 'ETHUSDT',
                    'action': 'SELL',
                    'entry_price': 3100.0,
                    'exit_price': 3050.0,
                    'position_size': 0.1,
                    'pnl': -5.0,
                    'confidence': 0.72,
                    'reason': 'Resistance level reached',
                    'timestamp': datetime.now(timezone.utc),
                    'status': 'completed'
                }
            ]
            
            for trade in sample_trades:
                await self.insert_trade(trade)
            
            # Insert performance metrics
            await self.execute("""
            INSERT INTO performance (win_rate, total_pnl, average_pnl, best_trade, worst_trade, 
                                   total_trades, best_strategy, timestamp)
            VALUES (0.6, 2.0, 1.0, 7.0, -5.0, 2, 'Technical Analysis', ?)
            """)
            
            logger.info("✅ Sample data inserted")
            
        except Exception as e:
            logger.error(f"Failed to insert sample data: {e}")

# Global database instance
_db_instance = None

async def get_database() -> DuckDBAdapter:
    """Get database instance (singleton)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DuckDBAdapter()
        connected = await _db_instance.connect()
        if connected:
            # Insert sample data if tables are empty
            trades = await _db_instance.get_today_trades()
            if not trades:
                await _db_instance.insert_sample_data()
    return _db_instance