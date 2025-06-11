import asyncio
import aiohttp
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class TelegramNotifier:
    """
    Sistema de notificações via Telegram para o agente de trading.
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = logging.getLogger('telegram_notifier')
    
    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Envia uma mensagem via Telegram.
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                data = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': parse_mode
                }
                
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        self.logger.info("Telegram message sent successfully")
                        return True
                    else:
                        self.logger.error(f"Failed to send Telegram message: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    async def send_trade_notification(self, symbol: str, trade_data: Dict[str, Any]) -> bool:
        """
        Envia notificação de trade executado.
        """
        try:
            main_order = trade_data.get('main_order', {})
            signal = trade_data.get('signal', {})
            stop_info = trade_data.get('stop_info', {})
            
            action = signal.get('action', 'unknown').upper()
            confidence = signal.get('confidence', 0) * 100
            price = main_order.get('price', 0)
            amount = main_order.get('amount', 0)
            stop_price = stop_info.get('stop_price', 0)
            
            message = f"""
🚀 <b>TRADE EXECUTADO</b>

📊 <b>Símbolo:</b> {symbol}
📈 <b>Ação:</b> {action}
💰 <b>Preço:</b> ${price:,.2f}
📦 <b>Quantidade:</b> {amount:.6f}
🎯 <b>Confiança:</b> {confidence:.1f}%
🛡️ <b>Stop Loss:</b> ${stop_price:,.2f}

⏰ <b>Horário:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {str(e)}")
            return False
    
    async def send_error_notification(self, error_message: str) -> bool:
        """
        Envia notificação de erro.
        """
        try:
            message = f"""
⚠️ <b>ERRO DETECTADO</b>

🔴 <b>Erro:</b> {error_message}
⏰ <b>Horário:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Por favor, verifique os logs para mais detalhes.
            """
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending error notification: {str(e)}")
            return False
    
    async def send_health_alert(self, health_status: Dict[str, Any]) -> bool:
        """
        Envia alerta de saúde do sistema.
        """
        try:
            alerts = health_status.get('issues', [])
            if not alerts:
                return True
            
            alerts_text = '\n'.join([f"• {alert}" for alert in alerts])
            
            message = f"""
🚨 <b>ALERTA DE SAÚDE</b>

⚠️ <b>Problemas detectados:</b>
{alerts_text}

⏰ <b>Horário:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending health alert: {str(e)}")
            return False
    
    async def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Envia resumo diário das operações.
        """
        try:
            trades_count = summary.get('trades_executed', 0)
            success_rate = summary.get('success_rate', 0) * 100
            total_pnl = summary.get('total_pnl', 0)
            
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            
            message = f"""
📊 <b>RESUMO DIÁRIO</b>

🔢 <b>Trades Executados:</b> {trades_count}
📊 <b>Taxa de Sucesso:</b> {success_rate:.1f}%
{pnl_emoji} <b>PnL Total:</b> ${total_pnl:,.2f}

⏰ <b>Data:</b> {datetime.now().strftime('%Y-%m-%d')}
            """
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {str(e)}")
            return False