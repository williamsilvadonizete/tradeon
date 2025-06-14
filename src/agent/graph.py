from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import sys
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, TypedDict, List, Optional
from datetime import datetime

# Adiciona o diretório raiz ao path para importar configurações
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE, MAX_TOKENS, CONFIDENCE_THRESHOLD,
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT
)

class AnalysisState(TypedDict):
    indicators: str
    analysis: Dict[str, Any]
    decision: Dict[str, Any]

class TradingAgent:
    def __init__(self):
        self.conn = self._get_db_connection()
        self._ensure_table()
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=OPENAI_API_KEY
        )
        
        # Define o prompt para análise
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em análise técnica de criptomoedas.
            Analise os indicadores técnicos fornecidos e forneça uma recomendação de trading.
            Considere:
            1. Força do sinal (-1 a 1)
            2. Força da tendência (0 a 1)
            3. Confiança na recomendação (0 a 1)
            4. Justificativa para a recomendação
            
            Responda em formato JSON com os campos:
            {
                "signal": float,  # -1 a 1
                "trend_strength": float,  # 0 a 1
                "confidence": float,  # 0 a 1
                "justification": string
            }"""),
            ("user", "{indicators}")
        ])
        
        # Define o prompt para decisão final
        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um gestor de risco de trading.
            Avalie a recomendação do analista e decida se deve executar a operação.
            Considere:
            1. Força do sinal
            2. Confiança na recomendação
            3. Força da tendência
            4. Histórico recente de operações
            
            Responda em formato JSON com os campos:
            {
                "execute": boolean,
                "action": string,  # "buy", "sell", ou "hold"
                "confidence": float,  # 0 a 1
                "reason": string
            }"""),
            ("user", "{analysis}")
        ])
        
        # Cria o grafo de decisão
        self.graph = self._create_decision_graph()

    def _get_db_connection(self):
        """Estabelece conexão com o PostgreSQL com fallback para modo offline"""
        try:
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                connect_timeout=10  # Timeout de 10 segundos
            )
            print("✅ Conectado ao PostgreSQL com sucesso")
            return conn
        except Exception as e:
            print(f"⚠️ Falha na conexão PostgreSQL: {str(e)}")
            print("🔄 Iniciando modo offline - usando fallback local")
            # Retorna None para indicar modo offline
            return None
        
    def _ensure_table(self):
        """Cria a tabela de trades se não existir (modo offline-safe)"""
        if self.conn is None:
            # Modo offline - cria estrutura em memória se necessário
            print("📱 Modo offline: usando armazenamento local")
            self.local_trades = []
            self.local_patterns = []
            return
            
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20),
                        action VARCHAR(10),
                        entry_price DECIMAL(20,8),
                        exit_price DECIMAL(20,8),
                        position_size DECIMAL(20,8),
                        pnl DECIMAL(20,8),
                        confidence DECIMAL(5,4),
                        reason TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.conn.commit()
                print("✅ Tabelas do banco de dados criadas/verificadas")
        except Exception as e:
            print(f"⚠️ Erro ao criar tabelas (modo offline ativo): {str(e)}")
            self.local_trades = []
            self.local_patterns = []

    def _create_decision_graph(self) -> StateGraph:
        """
        Cria o grafo de decisão para o processo de trading.
        
        Returns:
            StateGraph: Grafo LangGraph configurado
        """
        # Cria o grafo
        graph = StateGraph(AnalysisState)
        
        # Define a função de análise
        def analyze(state: AnalysisState) -> AnalysisState:
            chain = self.analysis_prompt | self.llm
            result = chain.invoke({"indicators": state["indicators"]})
            state["analysis"] = json.loads(result.content)
            return state
        
        # Define a função de decisão
        def decide(state: AnalysisState) -> AnalysisState:
            chain = self.decision_prompt | self.llm
            result = chain.invoke({"analysis": json.dumps(state["analysis"])})
            state["decision"] = json.loads(result.content)
            return state
            
        # Define a função final
        def final(state: AnalysisState) -> AnalysisState:
            # Aqui podemos fazer qualquer processamento final necessário
            return state
        
        # Adiciona os nós
        graph.add_node("analyze", analyze)
        graph.add_node("decide", decide)
        graph.add_node("final", final)
        
        # Define as conexões
        graph.add_edge("analyze", "decide")
        graph.add_edge("decide", "final")
        
        # Define o nó inicial e final
        graph.set_entry_point("analyze")
        graph.set_finish_point("final")
        
        return graph.compile()
        
    def analyze_market(self, indicators_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa os dados do mercado e gera uma recomendação.
        
        Args:
            indicators_data (dict): Dicionário com indicadores técnicos
            
        Returns:
            dict: Recomendação de trading
        """
        # Prepara os dados para o prompt
        indicators_str = json.dumps(indicators_data, indent=2)
        
        # Executa a análise
        result = self.graph.invoke({"indicators": indicators_str})
        
        # Processa o resultado
        try:
            decision = result["decision"]
            
            # Verifica se a confiança é suficiente
            if decision["confidence"] < CONFIDENCE_THRESHOLD:
                decision["execute"] = False
                decision["action"] = "hold"
                decision["reason"] = "Confiança insuficiente para executar a operação"
                
            return decision
            
        except (KeyError, json.JSONDecodeError):
            return {
                "execute": False,
                "action": "hold",
                "confidence": 0,
                "reason": "Erro ao processar a análise"
            }
            
    def update_memory(self, trade_result: Dict[str, Any]) -> None:
        """
        Atualiza a memória do agente com o resultado de uma operação.
        Args:
            trade_result (dict): Resultado da operação
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trades (symbol, action, entry_price, exit_price, position_size, pnl, confidence, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    trade_result.get("symbol"),
                    trade_result.get("action"),
                    trade_result.get("entry_price"),
                    trade_result.get("exit_price"),
                    trade_result.get("position_size"),
                    trade_result.get("pnl"),
                    trade_result.get("confidence"),
                    trade_result.get("reason"),
                ]
            )
            self.conn.commit()

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Consulta o histórico de trades (modo offline-safe).
        Args:
            limit (int): Número máximo de trades a retornar
        Returns:
            list[dict]: Lista de trades
        """
        if self.conn is None:
            # Modo offline - retorna dados mock
            return getattr(self, 'local_trades', [
                {'symbol': 'BTCUSDT', 'pnl': 100.0, 'timestamp': datetime.now()},
                {'symbol': 'ETHUSDT', 'pnl': -50.0, 'timestamp': datetime.now()},
                {'symbol': 'BTCUSDT', 'pnl': 75.0, 'timestamp': datetime.now()}
            ])[:limit]
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT %s",
                    [limit]
                )
                return cur.fetchall()
        except Exception as e:
            print(f"⚠️ Erro ao consultar histórico: {str(e)} - usando dados locais")
            return getattr(self, 'local_trades', [])[:limit]

class MemoryGraph(TradingAgent):
    """
    Extensão do TradingAgent que adiciona funcionalidades de memória e análise histórica.
    """
    def __init__(self):
        super().__init__()
        self._ensure_memory_tables()
        
    def _ensure_memory_tables(self):
        """Cria tabelas adicionais para armazenamento de memória e análise (modo offline-safe)"""
        if self.conn is None:
            print("📱 Modo offline: configurando armazenamento de memória local")
            self.local_market_memory = []
            self.local_pattern_memory = []
            return
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS market_memory (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indicators JSONB,
                    analysis JSONB,
                    decision JSONB,
                    success BOOLEAN,
                    pnl DECIMAL(20,8)
                )
            """)
            
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_memory (
                    id SERIAL PRIMARY KEY,
                    pattern_type VARCHAR(50),
                    pattern_data JSONB,
                    success_rate DECIMAL(5,4),
                    avg_pnl DECIMAL(20,8),
                    last_seen TIMESTAMP,
                    occurrences INTEGER DEFAULT 1
                    )
                """)
                self.conn.commit()
                print("✅ Tabelas de memória criadas/verificadas")
        except Exception as e:
            print(f"⚠️ Erro ao criar tabelas de memória (modo offline ativo): {str(e)}")
            self.local_market_memory = []
            self.local_pattern_memory = []
        
    def store_market_state(self, symbol: str, indicators: Dict[str, Any], 
                          analysis: Dict[str, Any], decision: Dict[str, Any]) -> None:
        """Armazena o estado atual do mercado na memória (modo offline-safe)."""
        if self.conn is None:
            # Modo offline - armazena localmente
            self.local_market_memory.append({
                'symbol': symbol,
                'indicators': indicators,
                'analysis': analysis,
                'decision': decision,
                'timestamp': datetime.now()
            })
            return
            
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO market_memory (symbol, indicators, analysis, decision)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [symbol, json.dumps(indicators), json.dumps(analysis), json.dumps(decision)]
                )
                self.conn.commit()
        except Exception as e:
            print(f"⚠️ Erro ao armazenar estado (usando local): {str(e)}")
            if not hasattr(self, 'local_market_memory'):
                self.local_market_memory = []
            self.local_market_memory.append({
                'symbol': symbol,
                'indicators': indicators,
                'analysis': analysis,
                'decision': decision,
                'timestamp': datetime.now()
            })
        
    def update_pattern_memory(self, pattern_type: str, pattern_data: Dict[str, Any], 
                            success: bool, pnl: float) -> None:
        """Atualiza a memória de padrões com novos resultados (modo offline-safe)."""
        if self.conn is None:
            # Modo offline - atualiza localmente
            if not hasattr(self, 'local_pattern_memory'):
                self.local_pattern_memory = []
            
            # Procura padrão existente
            for pattern in self.local_pattern_memory:
                if pattern['pattern_type'] == pattern_type:
                    # Atualiza estatísticas
                    pattern['occurrences'] += 1
                    pattern['success_rate'] = (pattern['success_rate'] * (pattern['occurrences'] - 1) + float(success)) / pattern['occurrences']
                    pattern['avg_pnl'] = (pattern['avg_pnl'] * (pattern['occurrences'] - 1) + pnl) / pattern['occurrences']
                    pattern['last_seen'] = datetime.now()
                    return
            
            # Novo padrão
            self.local_pattern_memory.append({
                'pattern_type': pattern_type,
                'pattern_data': pattern_data,
                'success_rate': float(success),
                'avg_pnl': pnl,
                'last_seen': datetime.now(),
                'occurrences': 1
            })
            return
            
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pattern_memory (pattern_type, pattern_data, success_rate, avg_pnl, last_seen)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (pattern_type) DO UPDATE
                    SET success_rate = (pattern_memory.success_rate * pattern_memory.occurrences + %s) / (pattern_memory.occurrences + 1),
                        avg_pnl = (pattern_memory.avg_pnl * pattern_memory.occurrences + %s) / (pattern_memory.occurrences + 1),
                        last_seen = CURRENT_TIMESTAMP,
                        occurrences = pattern_memory.occurrences + 1
                    """,
                    [pattern_type, json.dumps(pattern_data), float(success), pnl, 
                     datetime.now(), float(success), pnl]
                )
                self.conn.commit()
        except Exception as e:
            print(f"⚠️ Erro ao atualizar padrões (usando local): {str(e)}")
            self.update_pattern_memory(pattern_type, pattern_data, success, pnl)  # Fallback para modo offline
        
    def get_market_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Recupera o histórico de estados do mercado para um símbolo (modo offline-safe)."""
        if self.conn is None:
            # Modo offline - retorna dados locais
            local_data = getattr(self, 'local_market_memory', [])
            filtered = [item for item in local_data if item.get('symbol') == symbol]
            return sorted(filtered, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)[:limit]
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM market_memory 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                    """,
                    [symbol, limit]
                )
                return cur.fetchall()
        except Exception as e:
            print(f"⚠️ Erro ao consultar histórico de mercado: {str(e)} - usando dados locais")
            local_data = getattr(self, 'local_market_memory', [])
            filtered = [item for item in local_data if item.get('symbol') == symbol]
            return sorted(filtered, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)[:limit]
        
    def get_successful_patterns(self, min_success_rate: float = 0.6) -> List[Dict[str, Any]]:
        """Recupera padrões com taxa de sucesso acima do mínimo especificado (modo offline-safe)."""
        if self.conn is None:
            # Modo offline - retorna dados mock/locais
            local_patterns = getattr(self, 'local_pattern_memory', [
                {'pattern_type': 'bullish_momentum', 'success_rate': 0.72, 'avg_pnl': 85.5},
                {'pattern_type': 'trend_following', 'success_rate': 0.68, 'avg_pnl': 62.3},
                {'pattern_type': 'volume_breakout', 'success_rate': 0.65, 'avg_pnl': 48.7}
            ])
            return [p for p in local_patterns if p.get('success_rate', 0) >= min_success_rate]
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM pattern_memory 
                    WHERE success_rate >= %s 
                    ORDER BY success_rate DESC
                    """,
                    [min_success_rate]
                )
                return cur.fetchall()
        except Exception as e:
            print(f"⚠️ Erro ao consultar padrões: {str(e)} - usando dados locais")
            local_patterns = getattr(self, 'local_pattern_memory', [])
            return [p for p in local_patterns if p.get('success_rate', 0) >= min_success_rate]
        
    def analyze_market_with_memory(self, symbol: str, indicators_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa o mercado considerando o histórico armazenado na memória.
        """
        # Obtém análise básica
        decision = super().analyze_market(indicators_data)
        
        # Consulta padrões similares
        similar_patterns = self.get_successful_patterns()
        
        # Ajusta a decisão baseado no histórico
        if similar_patterns:
            avg_success_rate = sum(p['success_rate'] for p in similar_patterns) / len(similar_patterns)
            if avg_success_rate > 0.7 and decision['confidence'] < 0.8:
                decision['confidence'] = min(decision['confidence'] * 1.2, 0.95)
                decision['reason'] += f" (Padrão histórico com {avg_success_rate:.2%} de sucesso)"
        
        # Armazena o estado atual
        self.store_market_state(symbol, indicators_data, decision, decision)
        
        return decision 