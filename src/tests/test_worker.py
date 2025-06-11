import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agent.worker import RobustTradingAgent, AgentState


class TestRobustTradingAgent:
    """Testes para o agente de trading robusto."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.agent = RobustTradingAgent()
    
    def test_agent_initialization(self):
        """Testa a inicialização do agente."""
        assert self.agent.state == AgentState.INACTIVE
        assert not self.agent.running
        assert self.agent.metrics['trades_executed'] == 0
        assert self.agent.metrics['successful_trades'] == 0
        assert self.agent.metrics['total_pnl'] == 0.0
    
    def test_logger_setup(self):
        """Testa se o logger foi configurado corretamente."""
        assert self.agent.logger is not None
        assert self.agent.logger.name == 'robust_trading_agent'
        assert len(self.agent.logger.handlers) >= 2  # File + Console handlers
    
    @patch('src.agent.worker.EXCHANGE_API_KEY', 'test_key')
    @patch('src.agent.worker.EXCHANGE_SECRET', 'test_secret')
    @patch('src.exchanges.bybit.BybitAdapter')
    @patch('src.exchanges.order_manager.OrderManager')
    @patch('src.strategies.technical_analysis.TechnicalAnalysisStrategy')
    def test_initialize_components_success(self, mock_strategy, mock_order_manager, mock_exchange):
        """Testa inicialização bem-sucedida dos componentes."""
        # Mock da exchange
        mock_exchange_instance = Mock()
        mock_exchange.return_value = mock_exchange_instance
        
        # Mock do order manager
        mock_order_manager_instance = Mock()
        mock_order_manager.return_value = mock_order_manager_instance
        
        # Mock da estratégia
        mock_strategy_instance = Mock()
        mock_strategy.return_value = mock_strategy_instance
        
        # Executa inicialização
        result = self.agent.initialize_components()
        
        # Verificações
        assert result is True
        assert self.agent.state == AgentState.ACTIVE
        assert self.agent.exchange == mock_exchange_instance
        assert self.agent.order_manager == mock_order_manager_instance
        assert self.agent.strategy == mock_strategy_instance
        
        # Verifica se os métodos foram chamados
        mock_exchange_instance.initialize.assert_called_once()
    
    @patch('src.agent.worker.EXCHANGE_API_KEY', None)
    @patch('src.agent.worker.EXCHANGE_SECRET', None)
    def test_initialize_components_no_credentials(self):
        """Testa inicialização sem credenciais (modo demo)."""
        result = self.agent.initialize_components()
        
        assert result is True
        assert self.agent.state == AgentState.ACTIVE
        assert self.agent.exchange is None
    
    @patch('src.agent.worker.EXCHANGE_API_KEY', 'test_key')
    @patch('src.agent.worker.EXCHANGE_SECRET', 'test_secret')
    @patch('src.exchanges.bybit.BybitAdapter')
    def test_initialize_components_failure(self, mock_exchange):
        """Testa falha na inicialização dos componentes."""
        # Mock da exchange que levanta exceção
        mock_exchange.side_effect = Exception("Connection failed")
        
        # Executa inicialização
        result = self.agent.initialize_components()
        
        # Verificações
        assert result is False
        assert self.agent.state == AgentState.ERROR
        assert self.agent.metrics['errors'] == 1
        assert self.agent.metrics['last_error'] == "Connection failed"
    
    def test_analyze_and_trade_no_exchange(self):
        """Testa análise sem exchange disponível."""
        symbol = "BTCUSDT"
        
        # Executa análise sem exchange
        self.agent.analyze_and_trade(symbol)
        
        # Verifica que nenhum erro foi levantado
        assert self.agent.exchange is None
    
    @patch('src.agent.worker.EXCHANGE_API_KEY', 'test_key')
    @patch('src.agent.worker.EXCHANGE_SECRET', 'test_secret')
    def test_analyze_and_trade_with_data(self):
        """Testa análise com dados disponíveis."""
        # Setup mocks
        mock_exchange = Mock()
        mock_strategy = Mock()
        
        # Mock dos dados
        import pandas as pd
        mock_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        mock_exchange.get_historical_data.return_value = mock_data
        mock_strategy.calculate_indicators.return_value = mock_data
        mock_strategy.generate_signal.return_value = {
            'action': 'buy',
            'confidence': 0.8,
            'reason': ['Test signal']
        }
        mock_strategy.execute_trade.return_value = {'trade_id': '123'}
        
        self.agent.exchange = mock_exchange
        self.agent.strategy = mock_strategy
        
        # Executa análise
        symbol = "BTCUSDT"
        self.agent.analyze_and_trade(symbol)
        
        # Verificações
        assert symbol in self.agent.last_analysis
        assert self.agent.metrics['trades_executed'] == 1
        assert self.agent.metrics['successful_trades'] == 1  # confiança > 0.8
        
        # Verifica chamadas dos métodos
        mock_exchange.get_historical_data.assert_called_once()
        mock_strategy.calculate_indicators.assert_called_once()
        mock_strategy.generate_signal.assert_called_once()
        mock_strategy.execute_trade.assert_called_once()
    
    def test_analyze_and_trade_hold_signal(self):
        """Testa análise que resulta em sinal de hold."""
        # Setup mocks
        mock_exchange = Mock()
        mock_strategy = Mock()
        
        import pandas as pd
        mock_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        mock_exchange.get_historical_data.return_value = mock_data
        mock_strategy.calculate_indicators.return_value = mock_data
        mock_strategy.generate_signal.return_value = {
            'action': 'hold',
            'confidence': 0.5,
            'reason': ['No clear signal']
        }
        
        self.agent.exchange = mock_exchange
        self.agent.strategy = mock_strategy
        
        # Executa análise
        symbol = "BTCUSDT"
        self.agent.analyze_and_trade(symbol)
        
        # Verificações - nenhum trade deve ser executado
        assert self.agent.metrics['trades_executed'] == 0
        mock_strategy.execute_trade.assert_not_called()
    
    def test_get_status(self):
        """Testa obtenção do status do agente."""
        status = self.agent.get_status()
        
        # Verificações
        assert 'state' in status
        assert 'running' in status
        assert 'metrics' in status
        assert 'last_analysis' in status
        assert 'positions' in status
        assert 'timestamp' in status
        
        assert status['state'] == self.agent.state.value
        assert status['running'] == self.agent.running
        assert status['metrics'] == self.agent.metrics
    
    def test_is_running_states(self):
        """Testa verificação de estado de execução."""
        # Estado inicial
        assert not self.agent.is_running()
        
        # Estado ativo
        self.agent.running = True
        self.agent.state = AgentState.ACTIVE
        assert self.agent.is_running()
        
        # Estado de erro
        self.agent.state = AgentState.ERROR
        assert not self.agent.is_running()
        
        # Estado parando
        self.agent.state = AgentState.STOPPING
        assert not self.agent.is_running()
    
    def test_stop_agent(self):
        """Testa parada do agente."""
        # Inicia o agente
        self.agent.running = True
        self.agent.state = AgentState.ACTIVE
        
        # Para o agente
        self.agent.stop()
        
        # Verificações
        assert not self.agent.running
        assert self.agent.state == AgentState.STOPPING
    
    def test_stop_agent_not_running(self):
        """Testa parada do agente quando não está rodando."""
        # Agente não está rodando
        assert not self.agent.running
        
        # Tenta parar
        self.agent.stop()  # Não deve levantar exceção
        
        # Estado deve permanecer o mesmo
        assert not self.agent.running


@pytest.fixture
def mock_settings():
    """Fixture para configurações mock."""
    return {
        'EXCHANGE_API_KEY': 'test_key',
        'EXCHANGE_SECRET': 'test_secret',
        'TESTNET': True,
        'TRADING_PAIRS': ['BTCUSDT', 'ETHUSDT'],
        'TIMEFRAME': '1h',
        'WORKER_INTERVAL': 60
    }


class TestAgentIntegration:
    """Testes de integração para o agente."""
    
    @patch('src.agent.worker.EXCHANGE_API_KEY', 'test_key')
    @patch('src.agent.worker.EXCHANGE_SECRET', 'test_secret')
    @patch('src.agent.worker.TRADING_PAIRS', ['BTCUSDT'])
    def test_full_trading_cycle(self):
        """Testa um ciclo completo de trading (mock)."""
        agent = RobustTradingAgent()
        
        # Mock completo da cadeia
        with patch('src.exchanges.bybit.BybitAdapter') as mock_exchange_class, \
             patch('src.exchanges.order_manager.OrderManager') as mock_order_manager_class, \
             patch('src.strategies.technical_analysis.TechnicalAnalysisStrategy') as mock_strategy_class:
            
            # Setup mocks
            mock_exchange = Mock()
            mock_exchange_class.return_value = mock_exchange
            
            mock_order_manager = Mock()
            mock_order_manager_class.return_value = mock_order_manager
            
            mock_strategy = Mock()
            mock_strategy_class.return_value = mock_strategy
            
            import pandas as pd
            mock_data = pd.DataFrame({
                'close': [45000, 45100, 45200],
                'high': [45100, 45200, 45300],
                'low': [44900, 45000, 45100],
                'volume': [100, 110, 120]
            })
            
            mock_exchange.get_historical_data.return_value = mock_data
            mock_strategy.calculate_indicators.return_value = mock_data
            mock_strategy.generate_signal.return_value = {
                'action': 'buy',
                'confidence': 0.85,
                'reason': ['Strong buy signal']
            }
            mock_strategy.execute_trade.return_value = {
                'trade_id': 'test_123',
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'amount': 0.001
            }
            
            # Inicializa componentes
            result = agent.initialize_components()
            assert result is True
            
            # Executa análise para um símbolo
            agent.analyze_and_trade('BTCUSDT')
            
            # Verificações
            assert agent.metrics['trades_executed'] == 1
            assert agent.metrics['successful_trades'] == 1
            assert 'BTCUSDT' in agent.last_analysis
            
            # Verifica que todas as chamadas foram feitas
            mock_exchange.get_historical_data.assert_called()
            mock_strategy.calculate_indicators.assert_called()
            mock_strategy.generate_signal.assert_called()
            mock_strategy.execute_trade.assert_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])