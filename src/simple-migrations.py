#!/usr/bin/env python3
import os
import psycopg2
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configura as variáveis de ambiente do banco
os.environ['POSTGRES_HOST'] = 'crypto-trader-production-db.c63wya6gysbz.us-east-1.rds.amazonaws.com'
os.environ['POSTGRES_PORT'] = '5432'
os.environ['POSTGRES_DB'] = 'cryptotrader'
os.environ['POSTGRES_USER'] = 'cryptotrader'
os.environ['POSTGRES_PASSWORD'] = 'Angra123'

def get_db_connection():
    """Estabelece conexão com o banco de dados PostgreSQL."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "trading_bot"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def create_tables():
    """Cria as tabelas necessárias no banco de dados."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Tabela de trades
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
        
        # Tabela de performance
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
        
        # Tabela de configurações
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
        
        # Tabela de logs
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            level VARCHAR(10) NOT NULL,
            message TEXT NOT NULL,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Índices
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_symbol ON performance(symbol);
        CREATE INDEX IF NOT EXISTS idx_performance_period ON performance(period_start, period_end);
        CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
        CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at);
        """)
        
        # Função para atualizar updated_at
        cur.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """)
        
        # Triggers para atualizar updated_at
        cur.execute("""
        CREATE TRIGGER update_trades_updated_at
            BEFORE UPDATE ON trades
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
            
        CREATE TRIGGER update_configurations_updated_at
            BEFORE UPDATE ON configurations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)
        
        conn.commit()
        logger.info("Database tables created successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating tables: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def insert_initial_configurations():
    """Insere configurações iniciais no banco de dados."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Configurações de trading
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
        logger.info("Initial configurations inserted successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting configurations: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    try:
        print("Creating database tables...")
        create_tables()
        print("Inserting initial configurations...")
        insert_initial_configurations()
        print("Database setup completed successfully!")
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        exit(1)