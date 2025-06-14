import os
import logging
import asyncio
from typing import Optional
from datetime import datetime, timedelta
import asyncpg
from dotenv import load_dotenv
from src.db.database import get_db_connection, execute_query

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_tables() -> bool:
    """Create database tables if they don't exist."""
    try:
        logger.info("Starting database migration...")
        
        # Drop existing tables and functions
        logger.info("Dropping existing tables and functions...")
        await execute_query("""
            DROP TABLE IF EXISTS worker_logs CASCADE;
            DROP TABLE IF EXISTS trades CASCADE;
            DROP TABLE IF EXISTS trade_stats CASCADE;
            DROP FUNCTION IF EXISTS clean_old_logs() CASCADE;
        """)
        logger.info("✅ Existing tables and functions dropped")
        
        # Create worker_logs table
        logger.info("Creating worker_logs table...")
        await execute_query("""
            CREATE TABLE worker_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20) NOT NULL,
                exchange_id VARCHAR(50) NOT NULL,
                analysis JSONB NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP + INTERVAL '7 days'
            )
        """)
        logger.info("✅ worker_logs table created")
        
        # Create trades table
        logger.info("Creating trades table...")
        await execute_query("""
            CREATE TABLE trades (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20) NOT NULL,
                exchange_id VARCHAR(50) NOT NULL,
                side VARCHAR(10) NOT NULL,
                price DECIMAL NOT NULL,
                amount DECIMAL NOT NULL,
                status VARCHAR(20) NOT NULL,
                order_id VARCHAR(100),
                analysis JSONB
            )
        """)
        logger.info("✅ trades table created")
        
        # Create trade_stats table
        logger.info("Creating trade_stats table...")
        await execute_query("""
            CREATE TABLE trade_stats (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20) NOT NULL,
                exchange_id VARCHAR(50) NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_profit DECIMAL DEFAULT 0,
                win_rate DECIMAL DEFAULT 0,
                average_profit DECIMAL DEFAULT 0,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ trade_stats table created")
        
        # Create function to clean old logs
        logger.info("Creating clean_old_logs function...")
        await execute_query("""
            CREATE OR REPLACE FUNCTION clean_old_logs()
            RETURNS trigger AS $$
            BEGIN
                DELETE FROM worker_logs WHERE expires_at < CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        logger.info("✅ clean_old_logs function created")
        
        # Create trigger to clean old logs
        logger.info("Creating clean_old_logs trigger...")
        await execute_query("""
            CREATE TRIGGER clean_old_logs_trigger
            AFTER INSERT ON worker_logs
            EXECUTE FUNCTION clean_old_logs();
        """)
        logger.info("✅ clean_old_logs trigger created")
        
        logger.info("✅ All tables and functions created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating tables: {str(e)}")
        return False

async def run_migrations() -> bool:
    """Run database migrations."""
    max_attempts = 5
    attempt = 1
    
    while attempt <= max_attempts:
        try:
            logger.info(f"🔄 Migration attempt {attempt}/{max_attempts}")
            
            # Get database connection
            pool = await get_db_connection()
            if not pool:
                raise Exception("Failed to get database connection")
            
            # Create tables
            success = await create_tables()
            if not success:
                raise Exception("Failed to create tables")
            
            logger.info("✅ Database migrations completed successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Migration attempt {attempt} failed: {str(e)}")
            attempt += 1
            if attempt <= max_attempts:
                logger.info(f"⏳ Waiting 2 seconds before next attempt...")
                await asyncio.sleep(2)
            else:
                logger.error(f"❌ Failed to verify migrations after {max_attempts} attempts: {str(e)}")
                return False

if __name__ == "__main__":
    asyncio.run(run_migrations()) 