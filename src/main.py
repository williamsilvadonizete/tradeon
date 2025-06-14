#!/usr/bin/env python3
"""
Ponto de entrada principal do sistema de trading
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from src.agent.unified_worker import UnifiedIntelligentWorker
from src.api.server import start_api_thread

def setup_logging():
    """Configura logging"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/main.log')
        ]
    )

async def main():
    """Função principal"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Inicia API
        logger.info("Starting API...")
        start_api_thread()
        
        # Inicia worker
        logger.info("Starting Unified Intelligent Worker...")
        worker = UnifiedIntelligentWorker()
        await worker.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
    finally:
        logger.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())