import sys
import os
import time
from datetime import datetime
import json

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.exchanges.bybit import BybitAdapter
from src.strategies.technical_analysis import TechnicalAnalysisStrategy
from src.agent.memory import TradingMemory
from src.risk.manager import RiskManager
from config.settings import SYMBOLS, TIMEFRAME

class TradingBot:
    def __init__(self):
        # Inicializa o adaptador da Bybit
        self.exchange = BybitAdapter()
        
        # Inicializa a estratégia
        self.strategy = TechnicalAnalysisStrategy(self.exchange)
        
        # Inicializa o gerenciador de risco
        self.risk_manager = RiskManager()
        
        # Inicializa a memória
        self.memory = TradingMemory()
        
    def run(self):
        """
        Executa o loop principal do bot de trading.
        """
        print("Iniciando bot de trading...")
        
        while True:
            try:
                for symbol in SYMBOLS:
                    # Coleta dados
                    df = self.exchange.get_historical_data(symbol, TIMEFRAME)
                    if df is None:
                        print(f"Erro ao coletar dados para {symbol}")
                        continue
                        
                    # Calcula indicadores e gera sinal
                    df = self.strategy.calculate_indicators(df)
                    if df is None:
                        print(f"Erro ao calcular indicadores para {symbol}")
                        continue
                        
                    # Obtém o preço atual
                    current_price = self.exchange.get_current_price(symbol)
                    if current_price is None:
                        print(f"Erro ao obter preço atual para {symbol}")
                        continue
                        
                    # Gera sinal de trading
                    signal = self.strategy.generate_signal(df)
                    
                    # Verifica stop loss e take profit para posições existentes
                    if symbol in self.risk_manager.positions:
                        sl_tp_check = self.risk_manager.check_stop_loss_take_profit(
                            symbol, current_price
                        )
                        if sl_tp_check:
                            pnl = self.risk_manager.close_position(symbol, current_price)
                            self.memory.add_trade({
                                'symbol': symbol,
                                'action': 'close',
                                'reason': sl_tp_check,
                                'entry_price': self.risk_manager.positions[symbol]['entry_price'],
                                'exit_price': current_price,
                                'pnl': pnl
                            })
                            
                    # Executa nova operação se recomendado
                    if signal['action'] != 'hold':
                        # Obtém o saldo disponível
                        available_balance = self.exchange.get_balance('USDT')
                        
                        # Calcula o tamanho da posição
                        position_size = self.strategy.get_position_size(
                            signal, current_price, available_balance
                        )
                        
                        # Executa a operação
                        trade_result = self.strategy.execute_trade(
                            symbol, signal, position_size
                        )
                        
                        if trade_result:
                            # Atualiza o gerenciador de risco
                            self.risk_manager.update_position(
                                symbol,
                                current_price,
                                position_size,
                                is_long=(signal['action'] == 'buy')
                            )
                            
                            # Registra a operação
                            self.memory.add_trade({
                                'symbol': symbol,
                                'action': signal['action'],
                                'entry_price': current_price,
                                'position_size': position_size,
                                'confidence': signal['confidence'],
                                'reason': signal['reason']
                            })
                            
                    # Atualiza memória e salva
                    self.memory.save_to_file()
                    
                # Aguarda o próximo ciclo
                time.sleep(60)  # 1 minuto
                
            except Exception as e:
                print(f"Erro no loop principal: {e}")
                time.sleep(60)  # Aguarda 1 minuto em caso de erro
                
if __name__ == "__main__":
    bot = TradingBot()
    bot.run() 