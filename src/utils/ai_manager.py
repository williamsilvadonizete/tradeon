"""
Gerenciador de IA para o sistema de trading
Integra LangGraph e OpenAI para análise avançada
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import openai
from langchain.graphs import Graph
from langchain.graphs.graph_document import GraphDocument
from langchain.graphs.graph_store import GraphStore

# Configuração OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class TradingAnalysis:
    """Resultado da análise de trading"""
    symbol: str
    action: str
    confidence: float
    reasoning: str
    indicators: Dict[str, Any]
    timestamp: datetime

class AIManager:
    """Gerenciador de IA para análise de trading"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.graph = self._initialize_graph()
        
    def _initialize_graph(self) -> Graph:
        """Inicializa o grafo de conhecimento"""
        # Entidades principais
        entities = {
            'market_condition': ['bullish', 'bearish', 'sideways'],
            'technical_indicator': ['rsi', 'macd', 'ema', 'bollinger_bands'],
            'trading_signal': ['buy', 'sell', 'hold'],
            'risk_level': ['low', 'medium', 'high'],
            'timeframe': ['short_term', 'medium_term', 'long_term']
        }
        
        # Relações entre entidades
        relations = [
            ('market_condition', 'influences', 'trading_signal'),
            ('technical_indicator', 'generates', 'trading_signal'),
            ('risk_level', 'affects', 'trading_signal'),
            ('timeframe', 'contextualizes', 'trading_signal'),
            ('market_condition', 'determines', 'risk_level')
        ]
        
        # Cria grafo
        graph = Graph()
        
        # Adiciona entidades e relações
        for entity_type, values in entities.items():
            for value in values:
                graph.add_node(entity_type, value)
        
        for source, relation, target in relations:
            graph.add_edge(source, relation, target)
        
        return graph
    
    async def analyze_with_openai(self, data: Dict) -> Optional[TradingAnalysis]:
        """
        Analisa dados de trading usando OpenAI
        
        Args:
            data (Dict): Dados de trading para análise
            
        Returns:
            Optional[TradingAnalysis]: Resultado da análise
        """
        try:
            # Constrói prompt
            prompt = self._build_analysis_prompt(data)
            
            # Configura parâmetros
            params = {
                'model': 'gpt-4',
                'temperature': 0.3,
                'max_tokens': 500,
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
            
            # Envia requisição
            response = await openai.ChatCompletion.acreate(
                messages=[{'role': 'user', 'content': prompt}],
                **params
            )
            
            # Processa resposta
            analysis = self._parse_openai_response(response, data)
            
            # Atualiza grafo com novo conhecimento
            self._update_knowledge_graph(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI analysis: {str(e)}")
            return None
    
    def _build_analysis_prompt(self, data: Dict) -> str:
        """Constrói prompt para análise"""
        return f"""
        Analise os seguintes dados de trading e forneça uma recomendação:
        
        Símbolo: {data['symbol']}
        Preço atual: {data['price']}
        Indicadores técnicos:
        - RSI: {data['indicators']['rsi']}
        - MACD: {data['indicators']['macd_signal']}
        - EMA: {data['indicators']['ema_trend']}
        - Bollinger Bands: {data['indicators']['bb_position']}
        - Volume: {'Alto' if data['indicators']['volume_surge'] else 'Normal'}
        
        Forneça uma análise detalhada incluindo:
        1. Condição do mercado
        2. Força dos sinais técnicos
        3. Nível de risco
        4. Recomendação de ação (buy/sell/hold)
        5. Confiança na recomendação (0-1)
        6. Raciocínio por trás da decisão
        
        Formato da resposta:
        {{
            "market_condition": "string",
            "technical_strength": "string",
            "risk_level": "string",
            "action": "string",
            "confidence": float,
            "reasoning": "string"
        }}
        """
    
    def _parse_openai_response(self, response: Dict, data: Dict) -> TradingAnalysis:
        """Processa resposta da OpenAI"""
        try:
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            return TradingAnalysis(
                symbol=data['symbol'],
                action=analysis['action'],
                confidence=analysis['confidence'],
                reasoning=analysis['reasoning'],
                indicators=data['indicators'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing OpenAI response: {str(e)}")
            return None
    
    def _update_knowledge_graph(self, analysis: TradingAnalysis):
        """Atualiza grafo com novo conhecimento"""
        try:
            # Adiciona novo nó de análise
            self.graph.add_node('analysis', analysis.symbol)
            
            # Adiciona relações
            self.graph.add_edge('analysis', 'generates', analysis.action)
            self.graph.add_edge('analysis', 'has_confidence', str(analysis.confidence))
            
            # Adiciona relações com indicadores
            for indicator, value in analysis.indicators.items():
                self.graph.add_edge('analysis', 'uses', indicator)
                self.graph.add_edge(indicator, 'influences', analysis.action)
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {str(e)}")
    
    def get_market_context(self, symbol: str) -> Dict:
        """
        Obtém contexto de mercado do grafo
        
        Args:
            symbol (str): Símbolo do ativo
            
        Returns:
            Dict: Contexto de mercado
        """
        try:
            # Busca nós relacionados
            nodes = self.graph.get_neighbors('analysis')
            
            context = {
                'market_condition': None,
                'technical_signals': [],
                'risk_level': None,
                'historical_actions': []
            }
            
            # Processa nós
            for node in nodes:
                if node.type == 'market_condition':
                    context['market_condition'] = node.value
                elif node.type == 'technical_indicator':
                    context['technical_signals'].append(node.value)
                elif node.type == 'risk_level':
                    context['risk_level'] = node.value
                elif node.type == 'trading_signal':
                    context['historical_actions'].append(node.value)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return {} 