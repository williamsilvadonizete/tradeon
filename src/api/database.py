from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import logging
from alembic import op
import sqlalchemy as sa
from alembic.config import Config
from alembic import command
import asyncpg

logger = logging.getLogger(__name__)

# Configuração do banco de dados
POSTGRES_USER = os.getenv('POSTGRES_USER', 'cryptotrader')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'Angra123')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'db')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'trading_bot')

# Log das configurações do banco (sem a senha)
logger.info(f"Database configuration:")
logger.info(f"  Host: {POSTGRES_HOST}")
logger.info(f"  Port: {POSTGRES_PORT}")
logger.info(f"  Database: {POSTGRES_DB}")
logger.info(f"  User: {POSTGRES_USER}")

# Constrói a URL do banco de dados
DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Cria o engine do SQLAlchemy
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

# Cria a sessão
SessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Cria a base para os modelos
Base = declarative_base()

async def get_db():
    """Retorna uma sessão do banco de dados."""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Inicializa o banco de dados criando as tabelas."""
    try:
        # Importa os modelos para que o SQLAlchemy os reconheça
        from src.api.models import Trade, WorkerLog, TradeStats
        
        # Cria as tabelas
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise

async def run_migrations():
    """Executa as migrações do banco de dados usando Alembic."""
    try:
        # Configurar Alembic
        alembic_cfg = Config("alembic.ini")
        
        # Executar migrações
        command.upgrade(alembic_cfg, "head")
        logger.info("✅ Database migrations completed successfully")
    except Exception as e:
        logger.error(f"❌ Error running database migrations: {e}")
        raise

async def create_alembic_version():
    """Cria a tabela de versão do Alembic se não existir."""
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS alembic_version (
                    version_num VARCHAR(32) NOT NULL,
                    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
                );
            """))
            logger.info("✅ Alembic version table created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating alembic version table: {e}")
        raise 