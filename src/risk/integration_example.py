"""
Integration Example for Advanced Risk Manager

This module demonstrates how to integrate the AdvancedRiskManager
with the existing trading system.

Author: Trading Bot System
Date: 2025-11-06
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.risk.advanced_risk_manager import AdvancedRiskManager, PositionInfo
from src.risk.manager import RiskManager  # Original risk manager
from src.analysis.technical import AdvancedTechnicalAnalysis
from config.settings import TRADING_PAIRS, MAX_POSITION_SIZE, MAX_DRAWDOWN

class TradingSystemIntegration:
    """
    Integration class that combines the Advanced Risk Manager
    with the existing trading system components.
    """
    
    def __init__(self, config: dict = None):
        """Initialize the integrated trading system."""
        self.config = config or {
            'max_position_size': MAX_POSITION_SIZE,
            'max_drawdown': MAX_DRAWDOWN,
            'kelly_safety_factor': 0.5,
            'max_portfolio_heat': 0.15,
            'max_correlation_exposure': 0.4
        }
        
        # Initialize risk managers
        self.advanced_risk_manager = AdvancedRiskManager(self.config)
        self.legacy_risk_manager = RiskManager(self.config)
        
        # Initialize technical analysis
        self.technical_analyzer = AdvancedTechnicalAnalysis(self.config)
        
        # State tracking
        self.portfolio_value = 100000  # Starting portfolio value
        self.current_positions = {}
        self.price_data_cache = {}
        self.trade_history = []
        
        print("Trading System Integration initialized successfully")
    
    def update_price_data(self, symbol: str, ohlcv_data: pd.DataFrame):
        """Update price data cache for a symbol."""
        self.price_data_cache[symbol] = ohlcv_data.copy()
        
        # Add technical indicators
        self.price_data_cache[symbol] = self.technical_analyzer.calculate_indicators(
            self.price_data_cache[symbol]
        )
    
    def calculate_position_size(self, symbol: str, signal_strength: float) -> dict:
        """
        Calculate position size using both legacy and advanced risk managers.
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength from strategy (-1 to 1)
            
        Returns:
            Dictionary with position sizing information
        """
        if symbol not in self.price_data_cache:
            return {
                'error': 'No price data available for symbol',
                'symbol': symbol,
                'position_size': 0.0
            }
        
        # Get advanced risk manager recommendation
        advanced_result = self.advanced_risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=signal_strength,
            portfolio_value=self.portfolio_value,
            price_data=self.price_data_cache,
            trade_history=self.trade_history
        )
        
        # Get legacy risk manager recommendation
        price_data = self.price_data_cache[symbol]
        current_volatility = price_data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Mock win rate and profit ratio for legacy system
        win_rate = 0.6 if len(self.trade_history) == 0 else \
                   len([t for t in self.trade_history if t.get('pnl', 0) > 0]) / len(self.trade_history)
        
        profits = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t.get('pnl', 0) < 0]
        profit_ratio = (sum(profits) / len(profits)) / (sum(losses) / len(losses)) if profits and losses else 2.0
        
        legacy_size = self.legacy_risk_manager.calculate_position_size(
            win_rate=win_rate,
            profit_ratio=profit_ratio,
            current_equity=self.portfolio_value,
            volatility=current_volatility
        )
        
        return {
            'symbol': symbol,
            'signal_strength': signal_strength,
            'advanced_recommendation': advanced_result,
            'legacy_recommendation': legacy_size,
            'final_position_size': advanced_result.get('position_size', 0.0),
            'confidence': 'high' if advanced_result.get('status') == 'success' else 'low',
            'timestamp': datetime.now().isoformat()
        }
    
    def open_position(self, symbol: str, side: str, size: float, entry_price: float) -> dict:
        """
        Open a new position and update risk tracking.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            
        Returns:
            Dictionary with position information
        """
        # Create position info
        position = PositionInfo(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            side=side,
            unrealized_pnl=0.0,
            entry_time=datetime.now(),
            volatility_at_entry=self._get_current_volatility(symbol),
            correlation_risk=0.0,
            heat_contribution=0.0
        )
        
        # Add to current positions
        self.current_positions[symbol] = position
        
        # Update risk manager positions
        position_list = list(self.current_positions.values())
        self.advanced_risk_manager.update_positions(position_list)
        
        print(f"Opened {side} position: {symbol} | Size: {size} | Price: {entry_price}")
        
        return {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'status': 'opened',
            'timestamp': datetime.now().isoformat()
        }
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'manual') -> dict:
        """
        Close a position and update trade history.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Dictionary with trade result
        """
        if symbol not in self.current_positions:
            return {'error': f'No open position for {symbol}'}
        
        position = self.current_positions[symbol]
        
        # Calculate PnL
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        # Update portfolio value
        self.portfolio_value += pnl
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_percentage': (pnl / (position.entry_price * position.size)) * 100,
            'duration': (datetime.now() - position.entry_time).total_seconds() / 3600,  # hours
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update trade history
        self.trade_history.append(trade)
        self.advanced_risk_manager.update_trade_history(trade)
        
        # Remove from current positions
        del self.current_positions[symbol]
        
        # Update risk manager positions
        position_list = list(self.current_positions.values())
        self.advanced_risk_manager.update_positions(position_list)
        
        print(f"Closed {position.side} position: {symbol} | PnL: ${pnl:.2f} | Reason: {reason}")
        
        return trade
    
    def check_risk_management(self) -> dict:
        """
        Comprehensive risk management check.
        
        Returns:
            Dictionary with risk assessment and recommendations
        """
        recommendations = []
        alerts = []
        
        # Get comprehensive risk metrics
        risk_metrics = self.advanced_risk_manager.get_risk_metrics(
            self.portfolio_value, self.price_data_cache
        )
        
        # Check each position for close recommendations
        positions_to_close = []
        for symbol, position in self.current_positions.items():
            if symbol in self.price_data_cache:
                current_price = self.price_data_cache[symbol]['close'].iloc[-1]
                close_recommendation = self.advanced_risk_manager.should_close_position(
                    position, current_price, self.portfolio_value
                )
                
                if close_recommendation['should_close']:
                    positions_to_close.append({
                        'symbol': symbol,
                        'reasons': close_recommendation['reasons'],
                        'urgency': close_recommendation['urgency']
                    })
        
        # Generate recommendations
        if positions_to_close:
            recommendations.append(f"Consider closing {len(positions_to_close)} positions due to risk concerns")
        
        if risk_metrics.portfolio_heat > 0.12:
            alerts.append(f"High portfolio heat: {risk_metrics.portfolio_heat:.2%}")
        
        if risk_metrics.current_drawdown > 0.03:
            alerts.append(f"Significant drawdown: {risk_metrics.current_drawdown:.2%}")
        
        if risk_metrics.average_correlation > 0.7:
            alerts.append(f"High correlation risk: {risk_metrics.average_correlation:.2%}")
        
        return {
            'risk_metrics': {
                'portfolio_value': self.portfolio_value,
                'total_exposure': risk_metrics.total_exposure,
                'portfolio_heat': risk_metrics.portfolio_heat,
                'current_drawdown': risk_metrics.current_drawdown,
                'average_correlation': risk_metrics.average_correlation,
                'market_regime': risk_metrics.market_regime,
                'volatility_regime': risk_metrics.volatility_regime
            },
            'positions_to_close': positions_to_close,
            'recommendations': recommendations,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol."""
        if symbol not in self.price_data_cache:
            return 0.02  # Default 2% volatility
        
        returns = self.price_data_cache[symbol]['close'].pct_change().dropna()
        if len(returns) < 20:
            return 0.02
        
        return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    
    def get_portfolio_summary(self) -> dict:
        """Get comprehensive portfolio summary."""
        total_position_value = sum(
            abs(pos.size * pos.current_price) for pos in self.current_positions.values()
        )
        
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.current_positions.values())
        
        return {
            'portfolio_value': self.portfolio_value,
            'cash_available': self.portfolio_value - total_position_value,
            'total_position_value': total_position_value,
            'unrealized_pnl': unrealized_pnl,
            'number_of_positions': len(self.current_positions),
            'total_trades': len(self.trade_history),
            'win_rate': len([t for t in self.trade_history if t.get('pnl', 0) > 0]) / max(len(self.trade_history), 1),
            'current_positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for pos in self.current_positions.values()
            ]
        }

def create_mock_price_data(symbol: str, days: int = 100, base_price: float = 50000) -> pd.DataFrame:
    """Create mock price data for testing."""
    np.random.seed(42)
    
    # Generate realistic price movements
    returns = np.random.randn(days) * 0.02
    prices = (1 + returns).cumprod() * base_price
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(days) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(days)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(days)) * 0.005),
        'close': prices,
        'volume': np.random.rand(days) * 1000
    })
    
    return data

async def main():
    """Main example function demonstrating the integration."""
    print("=== Advanced Risk Manager Integration Example ===\n")
    
    # Initialize the integrated system
    config = {
        'max_position_size': 0.1,
        'max_drawdown': 0.05,
        'kelly_safety_factor': 0.5,
        'max_portfolio_heat': 0.15,
        'max_correlation_exposure': 0.4
    }
    
    trading_system = TradingSystemIntegration(config)
    
    # Add mock price data for multiple symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    base_prices = {'BTCUSDT': 50000, 'ETHUSDT': 3000, 'ADAUSDT': 1.0}
    
    for symbol in symbols:
        price_data = create_mock_price_data(symbol, 100, base_prices[symbol])
        trading_system.update_price_data(symbol, price_data)
        print(f"Updated price data for {symbol}")
    
    print("\n=== Position Sizing Examples ===")
    
    # Calculate position sizes for different signals
    for symbol in symbols:
        signal_strength = np.random.uniform(0.5, 1.0)
        sizing_result = trading_system.calculate_position_size(symbol, signal_strength)
        
        print(f"\n{symbol} Position Sizing:")
        print(f"  Signal Strength: {signal_strength:.2f}")
        print(f"  Advanced Recommendation: {sizing_result['advanced_recommendation']['position_size']:.4f}")
        print(f"  Legacy Recommendation: {sizing_result['legacy_recommendation']:.4f}")
        print(f"  Final Size: {sizing_result['final_position_size']:.4f}")
        print(f"  Confidence: {sizing_result['confidence']}")
    
    print("\n=== Opening Positions ===")
    
    # Open some positions
    for symbol in symbols[:2]:  # Open positions for first 2 symbols
        current_price = trading_system.price_data_cache[symbol]['close'].iloc[-1]
        sizing_result = trading_system.calculate_position_size(symbol, 0.8)
        position_size = sizing_result['final_position_size']
        
        if position_size > 0.01:  # Only open if size is meaningful
            position_result = trading_system.open_position(
                symbol=symbol,
                side='long',
                size=position_size,
                entry_price=current_price
            )
            print(f"  {position_result}")
    
    print("\n=== Portfolio Summary ===")
    portfolio_summary = trading_system.get_portfolio_summary()
    print(json.dumps(portfolio_summary, indent=2, default=str))
    
    print("\n=== Risk Management Check ===")
    risk_check = trading_system.check_risk_management()
    print(json.dumps(risk_check, indent=2, default=str))
    
    print("\n=== Simulating Price Changes and Risk Monitoring ===")
    
    # Simulate some price changes
    for symbol in trading_system.current_positions.keys():
        # Simulate a price drop
        original_data = trading_system.price_data_cache[symbol].copy()
        new_price = original_data['close'].iloc[-1] * 0.92  # 8% drop
        
        # Update the last price point
        new_row = original_data.iloc[-1:].copy()
        new_row['close'] = new_price
        new_row['high'] = max(new_price, new_row['high'].iloc[0])
        new_row['low'] = min(new_price, new_row['low'].iloc[0])
        
        updated_data = pd.concat([original_data, new_row])
        trading_system.update_price_data(symbol, updated_data)
        
        print(f"Simulated price drop for {symbol}: {new_price:.2f}")
    
    print("\n=== Risk Check After Price Changes ===")
    risk_check_after = trading_system.check_risk_management()
    print(json.dumps(risk_check_after, indent=2, default=str))
    
    # Close positions if recommended
    if risk_check_after['positions_to_close']:
        print("\n=== Closing Positions Based on Risk Recommendations ===")
        for position_to_close in risk_check_after['positions_to_close']:
            symbol = position_to_close['symbol']
            current_price = trading_system.price_data_cache[symbol]['close'].iloc[-1]
            
            trade_result = trading_system.close_position(
                symbol=symbol,
                exit_price=current_price,
                reason=f"Risk management: {', '.join(position_to_close['reasons'])}"
            )
            print(f"  Closed {symbol}: PnL = ${trade_result['pnl']:.2f}")
    
    print("\n=== Final Portfolio Summary ===")
    final_portfolio = trading_system.get_portfolio_summary()
    print(json.dumps(final_portfolio, indent=2, default=str))
    
    print("\n=== Comprehensive Risk Report ===")
    risk_report = trading_system.advanced_risk_manager.export_risk_report(
        trading_system.portfolio_value, trading_system.price_data_cache
    )
    print(json.dumps(risk_report, indent=2, default=str))

if __name__ == "__main__":
    # Run the integration example
    asyncio.run(main())