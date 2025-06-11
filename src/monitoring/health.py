import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil
import os

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    uptime: timedelta

@dataclass
class BotMetrics:
    total_trades: int
    active_trades: int
    win_rate: float
    current_drawdown: float
    daily_pnl: float
    last_trade_time: datetime
    api_latency: float

class HealthMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.metrics_history: List[Dict] = []
        self.alert_thresholds = {
            'cpu_usage': self.config.get('cpu_threshold', 80.0),
            'memory_usage': self.config.get('memory_threshold', 80.0),
            'disk_usage': self.config.get('disk_threshold', 80.0),
            'api_latency': self.config.get('latency_threshold', 1000.0),  # ms
            'drawdown': self.config.get('drawdown_threshold', 0.1),  # 10%
        }
        
    def check_system_health(self) -> SystemMetrics:
        """
        Verifica a saúde do sistema.
        
        Returns:
            SystemMetrics com métricas do sistema
        """
        try:
            # CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memória
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Rede
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            
            # Uptime
            uptime = datetime.now() - self.start_time
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                uptime=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {str(e)}")
            return SystemMetrics(0.0, 0.0, 0.0, {}, timedelta())
            
    def check_bot_health(self, bot_metrics: Dict) -> BotMetrics:
        """
        Verifica a saúde do bot de trading.
        
        Args:
            bot_metrics: Dicionário com métricas do bot
            
        Returns:
            BotMetrics com métricas do bot
        """
        try:
            return BotMetrics(
                total_trades=bot_metrics.get('total_trades', 0),
                active_trades=bot_metrics.get('active_trades', 0),
                win_rate=bot_metrics.get('win_rate', 0.0),
                current_drawdown=bot_metrics.get('current_drawdown', 0.0),
                daily_pnl=bot_metrics.get('daily_pnl', 0.0),
                last_trade_time=bot_metrics.get('last_trade_time', datetime.now()),
                api_latency=bot_metrics.get('api_latency', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error checking bot health: {str(e)}")
            return BotMetrics(0, 0, 0.0, 0.0, 0.0, datetime.now(), 0.0)
            
    def check_health(self, bot_metrics: Dict) -> Dict:
        """
        Verifica a saúde geral do sistema e do bot.
        
        Args:
            bot_metrics: Dicionário com métricas do bot
            
        Returns:
            Dicionário com status de saúde e alertas
        """
        try:
            # Verifica sistema
            system_metrics = self.check_system_health()
            bot_metrics = self.check_bot_health(bot_metrics)
            
            # Verifica alertas
            alerts = []
            
            # CPU
            if system_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                alerts.append(f"High CPU usage: {system_metrics.cpu_usage:.1f}%")
                
            # Memória
            if system_metrics.memory_usage > self.alert_thresholds['memory_usage']:
                alerts.append(f"High memory usage: {system_metrics.memory_usage:.1f}%")
                
            # Disco
            if system_metrics.disk_usage > self.alert_thresholds['disk_usage']:
                alerts.append(f"High disk usage: {system_metrics.disk_usage:.1f}%")
                
            # API Latency
            if bot_metrics.api_latency > self.alert_thresholds['api_latency']:
                alerts.append(f"High API latency: {bot_metrics.api_latency:.1f}ms")
                
            # Drawdown
            if bot_metrics.current_drawdown > self.alert_thresholds['drawdown']:
                alerts.append(f"High drawdown: {bot_metrics.current_drawdown:.1%}")
                
            # Inatividade
            if (datetime.now() - bot_metrics.last_trade_time) > timedelta(hours=24):
                alerts.append("No trades in the last 24 hours")
                
            # Salva métricas no histórico
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'system_metrics': system_metrics,
                'bot_metrics': bot_metrics,
                'alerts': alerts
            })
            
            # Mantém apenas últimas 24 horas
            self.metrics_history = [
                m for m in self.metrics_history
                if datetime.now() - m['timestamp'] <= timedelta(hours=24)
            ]
            
            return {
                'status': 'healthy' if not alerts else 'unhealthy',
                'alerts': alerts,
                'system_metrics': system_metrics,
                'bot_metrics': bot_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error checking health: {str(e)}")
            return {
                'status': 'error',
                'alerts': [f"Health check error: {str(e)}"],
                'system_metrics': None,
                'bot_metrics': None
            }
            
    def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """
        Retorna o histórico de métricas.
        
        Args:
            hours: Número de horas de histórico
            
        Returns:
            Lista de métricas históricas
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m['timestamp'] >= cutoff]
        
    def get_health_summary(self) -> Dict:
        """
        Retorna um resumo da saúde do sistema.
        
        Returns:
            Dicionário com resumo
        """
        try:
            if not self.metrics_history:
                return {'status': 'unknown', 'message': 'No metrics available'}
                
            latest = self.metrics_history[-1]
            
            return {
                'status': latest['status'],
                'alerts': latest['alerts'],
                'uptime': str(latest['system_metrics'].uptime),
                'cpu_usage': latest['system_metrics'].cpu_usage,
                'memory_usage': latest['system_metrics'].memory_usage,
                'total_trades': latest['bot_metrics'].total_trades,
                'win_rate': latest['bot_metrics'].win_rate,
                'current_drawdown': latest['bot_metrics'].current_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health summary: {str(e)}")
            return {'status': 'error', 'message': str(e)} 