import unittest
from src.risk.manager import RiskManager, RiskMetrics
from datetime import datetime

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'max_drawdown': 0.1,
            'trailing_stop': 0.02,
            'kelly_fraction': 0.5,
            'max_position_size': 0.1,
            'min_position_size': 0.01,
            'risk_per_trade': 0.02
        }
        self.risk_manager = RiskManager(self.config)
        
    def test_calculate_position_size(self):
        # Teste com win rate e profit ratio ideais
        position_size = self.risk_manager.calculate_position_size(
            win_rate=0.6,
            profit_ratio=2.0,
            current_equity=10000,
            volatility=0.1
        )
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, self.config['max_position_size'])
        
        # Teste com volatilidade alta
        position_size_high_vol = self.risk_manager.calculate_position_size(
            win_rate=0.6,
            profit_ratio=2.0,
            current_equity=10000,
            volatility=0.5
        )
        self.assertLess(position_size_high_vol, position_size)
        
        # Teste com win rate baixo
        position_size_low_win = self.risk_manager.calculate_position_size(
            win_rate=0.3,
            profit_ratio=2.0,
            current_equity=10000,
            volatility=0.1
        )
        self.assertLess(position_size_low_win, position_size)
        
    def test_update_trailing_stop(self):
        # Teste para posição long
        current_price = 100
        highest_price = 110
        stop_long = self.risk_manager.update_trailing_stop(
            current_price=current_price,
            highest_price=highest_price,
            position_type='long'
        )
        self.assertEqual(stop_long, highest_price * (1 - self.config['trailing_stop']))
        
        # Teste para posição short
        stop_short = self.risk_manager.update_trailing_stop(
            current_price=current_price,
            highest_price=highest_price,
            position_type='short'
        )
        self.assertEqual(stop_short, highest_price * (1 + self.config['trailing_stop']))
        
    def test_check_drawdown(self):
        # Teste com drawdown aceitável
        self.risk_manager.metrics.peak_equity = 10000
        is_acceptable = self.risk_manager.check_drawdown(9500)
        self.assertTrue(is_acceptable)
        
        # Teste com drawdown excessivo
        is_acceptable = self.risk_manager.check_drawdown(8500)
        self.assertFalse(is_acceptable)
        
    def test_update_metrics(self):
        trade_result = {
            'total_trades': 100,
            'winning_trades': 60,
            'total_profit': 2000,
            'total_loss': 1000
        }
        
        self.risk_manager.update_metrics(trade_result)
        
        self.assertEqual(self.risk_manager.metrics.win_rate, 0.6)
        self.assertEqual(self.risk_manager.metrics.profit_ratio, 2.0)
        
    def test_get_risk_metrics(self):
        metrics = self.risk_manager.get_risk_metrics()
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.profit_ratio, 0.0)
        self.assertEqual(metrics.max_drawdown, 0.0)
        self.assertEqual(metrics.current_drawdown, 0.0)
        self.assertEqual(metrics.peak_equity, 0.0)
        self.assertEqual(metrics.current_equity, 0.0)

if __name__ == '__main__':
    unittest.main() 