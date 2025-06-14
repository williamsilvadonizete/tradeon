"""
Configurações centralizadas do sistema de trading
"""

import os
from typing import List

# Exchange settings
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'gateio')
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY', '')
EXCHANGE_SECRET = os.getenv('EXCHANGE_SECRET', '')
EXCHANGE_TESTNET = os.getenv('EXCHANGE_TESTNET', 'false').lower() == 'true'

# Trading parameters
TRADING_PAIRS: List[str] = os.getenv('TRADING_PAIRS', 'BTCUSDT,ETHUSDT').split(',')
TIMEFRAME = os.getenv('TIMEFRAME', '1h')
WORKER_INTERVAL = int(os.getenv('WORKER_INTERVAL', '60'))

# Technical analysis parameters
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
EMA_FAST = int(os.getenv('EMA_FAST', '9'))
EMA_SLOW = int(os.getenv('EMA_SLOW', '21'))
MACD_FAST = int(os.getenv('MACD_FAST', '12'))
MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))
BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
BB_STD = float(os.getenv('BB_STD', '2'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

# Risk management
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2% do capital por trade
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '3'))
STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.02'))  # 2%
TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.04'))  # 4%

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DIR = 'logs'

# API settings
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_DEBUG = os.getenv('API_DEBUG', 'false').lower() == 'true'

# Configurações de API
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

# Configurações de Exchange
EXCHANGE = "gateio"
MARKET_TYPE = "spot"  # spot ou futures
LEVERAGE = 5

# Configurações de Volume e Volatilidade
MIN_VOLUME = 1000000  # Volume mínimo em USDT
MIN_VOLATILITY = 0.001  # Volatilidade mínima (0.1%)
MAX_SPREAD = 0.01  # Spread máximo (1%)

# Configurações de Posição
MAX_POSITION_SIZE = 0.1  # 10% do capital
MIN_POSITION_SIZE = 0.01  # 1% do capital

# Configurações de Stop Loss e Take Profit
STOP_LOSS_PERCENT = 0.02  # 2%
TAKE_PROFIT_PERCENT = 0.04  # 4%

# Configurações de Banco de Dados
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "trading_bot"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# Configurações de Logging
LOG_FILE = "trading_bot.log"

# Configurações de Backtesting
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2023-12-31"
BACKTEST_INITIAL_CAPITAL = 10000  # USDT

# Configurações de Performance
PERFORMANCE_METRICS = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate"
]

# Configurações de Notificação
ENABLE_NOTIFICATIONS = True
NOTIFICATION_CHANNELS = [
    "email",
    "telegram"
]

# Configurações de Email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"

# Configurações de Telegram
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id" 