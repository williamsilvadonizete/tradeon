#!/usr/bin/env python3

import os
import sys
import time
import hashlib
import hmac
import requests
import json
import logging
from datetime import datetime

# Configura logging
logging.basicConfig(level=logging.INFO, format='🧪 %(asctime)s [%(levelname)s] gateio_test: %(message)s')
logger = logging.getLogger(__name__)

def test_gateio_connection():
    """
    Testa conexão com Gate.io API usando as credenciais do ambiente.
    """
    
    # Pega as credenciais das variáveis de ambiente
    api_key = os.getenv('EXCHANGE_API_KEY', '')
    api_secret = os.getenv('EXCHANGE_SECRET', '')
    testnet = os.getenv('EXCHANGE_TESTNET', 'false').lower() == 'true'
    
    logger.info(f"Testing Gate.io API connection...")
    logger.info(f"API Key: {api_key[:10]}..." if api_key else "API Key: MISSING")
    logger.info(f"API Secret: {'*' * 20}" if api_secret else "API Secret: MISSING")
    logger.info(f"Testnet mode: {testnet}")
    
    if not api_key or not api_secret:
        logger.error("❌ Credenciais não encontradas!")
        return False
    
    # Seleciona o host correto
    if testnet:
        host = "https://fx-api-testnet.gateio.ws"  # Testnet
        logger.info("🧪 Using TESTNET environment")
    else:
        host = "https://api.gateio.ws"  # Produção
        logger.info("🏭 Using PRODUCTION environment")
    
    # Endpoint para buscar saldo
    prefix = "/api/v4"
    endpoint = "/spot/accounts"
    url = f"{host}{prefix}{endpoint}"
    
    # Timestamp atual
    timestamp = str(int(time.time()))
    
    # Método e query string
    method = "GET"
    query_string = ""
    
    # Corpo da requisição (vazio para GET)
    body = ""
    
    # Cria a string para assinar
    message = f"{method}\n{prefix}{endpoint}\n{query_string}\n{hashlib.sha512(body.encode()).hexdigest()}\n{timestamp}"
    
    # Calcula a assinatura
    signature = hmac.new(
        api_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha512
    ).hexdigest()
    
    # Headers da requisição
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'KEY': api_key,
        'SIGN': signature,
        'Timestamp': timestamp
    }
    
    logger.info(f"🌐 Fazendo requisição para: {url}")
    logger.info(f"🔐 Timestamp: {timestamp}")
    logger.info(f"📝 Message to sign: {message[:100]}...")
    
    try:
        # Faz a requisição
        response = requests.get(url, headers=headers, timeout=15)
        
        logger.info(f"📊 Status Code: {response.status_code}")
        logger.info(f"📄 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Sucesso! Conexão com Gate.io estabelecida")
            logger.info(f"💰 Saldo da carteira ({len(data)} moedas):")
            
            total_value = 0
            currencies_with_balance = 0
            
            for account in data:
                currency = account.get('currency', 'Unknown')
                available = float(account.get('available', 0))
                locked = float(account.get('locked', 0))
                total = available + locked
                
                if total > 0:
                    currencies_with_balance += 1
                    logger.info(f"  💰 {currency}: {total:.8f} (Disponível: {available:.8f}, Bloqueado: {locked:.8f})")
                    
                    # Calcula valor em USDT aproximado (só para as principais)
                    if currency in ['USDT', 'USD']:
                        total_value += total
            
            logger.info(f"📊 Resumo: {currencies_with_balance} moedas com saldo > 0")
            if total_value > 0:
                logger.info(f"💵 Valor estimado em USDT: ${total_value:.2f}")
            
            return True
            
        elif response.status_code == 401:
            logger.error(f"❌ Erro de autenticação (401): Credenciais inválidas")
            logger.error(f"Response: {response.text}")
            return False
        elif response.status_code == 403:
            logger.error(f"❌ Erro de permissão (403): API sem permissão para spot trading")
            logger.error(f"Response: {response.text}")
            return False
        else:
            logger.error(f"❌ Erro HTTP {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Erro de conexão: {str(e)}")
        logger.error("Possíveis causas: Sem internet, DNS bloqueado, ou firewall")
        return False
    except requests.exceptions.Timeout as e:
        logger.error(f"❌ Timeout na requisição: {str(e)}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erro de requisição: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro geral: {str(e)}")
        return False

def test_basic_connectivity():
    """
    Testa conectividade básica com a internet.
    """
    logger.info(f"🌐 Testing basic connectivity...")
    
    try:
        # Testa DNS resolution e conectividade HTTP
        response = requests.get("https://httpbin.org/ip", timeout=10)
        if response.status_code == 200:
            ip_info = response.json()
            logger.info(f"✅ Internet connectivity OK. Public IP: {ip_info.get('origin', 'Unknown')}")
            return True
        else:
            logger.error(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ No internet connectivity: {str(e)}")
        return False

def test_gateio_public_api():
    """
    Testa API pública da Gate.io (sem autenticação).
    """
    logger.info(f"🌐 Testing Gate.io public API...")
    
    try:
        # Testa endpoint público de moedas
        response = requests.get("https://api.gateio.ws/api/v4/spot/currencies", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Gate.io public API OK. {len(data)} currencies available")
            return True
        else:
            logger.error(f"❌ Gate.io public API error: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gate.io public API error: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Gate.io API Connection Test")
    logger.info("=" * 60)
    
    # 1. Teste de conectividade básica
    connectivity_ok = test_basic_connectivity()
    
    if connectivity_ok:
        logger.info("")
        
        # 2. Teste da API pública Gate.io
        public_api_ok = test_gateio_public_api()
        
        if public_api_ok:
            logger.info("")
            
            # 3. Teste da API privada (com credenciais)
            auth_ok = test_gateio_connection()
            
            if auth_ok:
                logger.info("🎉 TODOS OS TESTES PASSARAM! Gate.io está funcionando")
            else:
                logger.error("❌ Falha na autenticação com Gate.io")
        else:
            logger.error("❌ Gate.io API não está acessível")
    else:
        logger.error("❌ Sem conectividade com a internet")
    
    logger.info("=" * 60)
    logger.info("🏁 Test completed")
    logger.info("=" * 60)