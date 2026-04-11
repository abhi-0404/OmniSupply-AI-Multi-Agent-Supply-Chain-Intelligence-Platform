"""
Base agent class for OmniSupply platform.
All specialized agents inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from opik import track
from opik.integrations.langchain import OpikTracer

from ..utils.retry import gemini_retry

# Only use OpikTracer if OPIK_API_KEY is configured
def _get_callbacks():
    if os.getenv("OPIK_API_KEY") or os.getenv("COMET_API_KEY"):
        return [OpikTracer(project_name=OPIK_PROJECT_NAME)]
    return []


def _worker_model() -> str:
    """Return the model name for worker agents.
    Prefers GEMINI_WORKER_MODEL, falls back to GEMINI_MODEL."""
    return os.getenv("GEMINI_WORKER_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

from ..data.models import AgentResult
from ..storage.sql.database import DatabaseClient
from ..storage.vector.chromadb_client import OmniSupplyVectorStore

logger = logging.getLogger(__name__)

# Get Opik project name from environment
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "omnisupply")


class BaseAgentState(TypedDict):
    """Base state for all agents"""
    session_id: str
    user_query: str
    context: Dict[str, Any]
    result: Optional[AgentResult]
    error: Optional[str]


class BaseAgent(ABC):
    """
    Abstract base class for all OmniSupply agents.

    All agents must implement:
    - _build_graph(): Define the LangGraph workflow
    - get_capabilities(): Return list of what the agent can do
    - _format_result(): Convert state to AgentResult
    """

    def __init__(
        self,
        name: str,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        db_client: Optional[DatabaseClient] = None,
        vector_store: Optional[OmniSupplyVectorStore] = None,
        **kwargs
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name (e.g., "data_analyst")
            llm: Language model instance
            db_client: Database connection
            vector_store: Vector store for semantic search
            **kwargs: Additional agent-specific config
        """
        self.name = name
        self.llm = llm or ChatGoogleGenerativeAI(
            model=_worker_model(),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            callbacks=_get_callbacks()
        )
        self.db = db_client
        self.vector_store = vector_store
        self.config = kwargs

        # Build LangGraph workflow
        self.graph = self._build_graph()

        logger.info(f"✅ {self.name} agent initialized")

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for this agent.

        Returns:
            Compiled StateGraph
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of agent capabilities.

        Returns:
            List of capability descriptions
        """
        pass

    @abstractmethod
    def _format_result(self, state: Dict[str, Any]) -> AgentResult:
        """
        Convert agent state to AgentResult.

        Args:
            state: Final agent state

        Returns:
            AgentResult with insights, metrics, recommendations
        """
        pass

    @track(project_name=OPIK_PROJECT_NAME)
    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> AgentResult:
        """
        Execute agent workflow.

        Args:
            query: User query or command
            context: Additional context (e.g., from other agents)
            session_id: Session ID for tracking

        Returns:
            AgentResult with insights and recommendations
        """
        start_time = datetime.now()

        logger.info(f"🤖 {self.name} executing: {query[:100]}")

        # Prepare initial state
        initial_state = self._prepare_state(query, context, session_id)

        try:
            # Execute graph
            # Execute graph with exponential-backoff retry on rate-limit errors
            final_state = gemini_retry(
                lambda: self.graph.invoke(
                    initial_state,
                    config={"callbacks": _get_callbacks()}
                ),
                max_retries=3,
                base_delay=5,
            )

            # Format result
            result = self._format_result(final_state)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time

            logger.info(f"✅ {self.name} completed in {execution_time:.0f}ms")

            return result

        except Exception as e:
            logger.error(f"❌ {self.name} failed: {e}")

            # Return error result
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return AgentResult(
                agent_name=self.name,
                query=query,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )

    def _prepare_state(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare initial agent state"""
        return {
            "session_id": session_id or f"{self.name}_{datetime.now().timestamp()}",
            "user_query": query,
            "context": context or {},
            "result": None,
            "error": None
        }

    def can_handle(self, query: str) -> float:
        """
        Determine if this agent can handle the query.

        Args:
            query: User query

        Returns:
            Confidence score 0.0-1.0
        """
        # Default implementation - check for keywords
        query_lower = query.lower()
        capabilities = [c.lower() for c in self.get_capabilities()]

        # Simple keyword matching
        matches = sum(1 for cap in capabilities if any(word in query_lower for word in cap.split()))

        return min(matches / len(capabilities), 1.0) if capabilities else 0.0

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "capabilities": self.get_capabilities(),
            "llm_model": self.llm.model_name if hasattr(self.llm, 'model_name') else "unknown",
            "has_db": self.db is not None,
            "has_vector_store": self.vector_store is not None
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"


class AgentRegistry:
    """Registry for all available agents"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        logger.info("Agent registry initialized")

    def register(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agents"""
        return list(self.agents.keys())

    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents"""
        return {
            name: agent.get_capabilities()
            for name, agent in self.agents.items()
        }

    def find_best_agent(self, query: str) -> Optional[BaseAgent]:
        """Find best agent for a query"""
        if not self.agents:
            return None

        scores = {
            name: agent.can_handle(query)
            for name, agent in self.agents.items()
        }

        best_agent_name = max(scores, key=scores.get)
        best_score = scores[best_agent_name]

        if best_score > 0.3:  # Confidence threshold
            logger.info(f"Best agent for query: {best_agent_name} (score: {best_score:.2f})")
            return self.agents[best_agent_name]

        logger.warning(f"No confident agent found (best score: {best_score:.2f})")
        return None

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        return f"<AgentRegistry agents={list(self.agents.keys())}>"
