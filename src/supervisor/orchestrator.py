"""
Supervisor Agent for OmniSupply platform.
Orchestrates multiple specialized agents to fulfill complex queries.

Rate-limit strategy:
  - Supervisor uses GEMINI_SUPERVISOR_MODEL (more capable, lower quota)
  - Worker agents use GEMINI_WORKER_MODEL (lighter, higher quota)
  - All agent calls are sequential with AGENT_CALL_DELAY_MS between them
  - gemini_retry() wraps every LLM call with exponential backoff on 429s
"""

from typing import Dict, Any, List, Optional, TypedDict, Literal
from datetime import datetime
import time
import logging
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from opik import track
from opik.integrations.langchain import OpikTracer

from ..agents.base import AgentRegistry
from ..data.models import AgentResult
from ..utils.retry import gemini_retry

logger = logging.getLogger(__name__)

OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "omnisupply")


def _supervisor_model() -> str:
    """Return the model for the Supervisor.
    Prefers GEMINI_SUPERVISOR_MODEL, falls back to GEMINI_MODEL."""
    return (
        os.getenv("GEMINI_SUPERVISOR_MODEL")
        or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )


def _agent_call_delay() -> float:
    """Return inter-agent delay in seconds (from AGENT_CALL_DELAY_MS, default 2000)."""
    return int(os.getenv("AGENT_CALL_DELAY_MS", "2000")) / 1000.0


def _get_callbacks():
    if os.getenv("OPIK_API_KEY") or os.getenv("COMET_API_KEY"):
        return [OpikTracer(project_name=OPIK_PROJECT_NAME)]
    return []


# ── Pydantic structured-output models ────────────────────────────────────────

class AgentSelection(BaseModel):
    agents: List[str] = Field(description="List of agent names to invoke")
    reasoning: str = Field(description="Why these agents were selected")
    execution_order: Literal['parallel', 'sequential'] = Field(
        description="How to execute agents (both treated as sequential with throttle)"
    )


class TaskPlan(BaseModel):
    steps: List[str] = Field(description="Step-by-step plan")
    agents_needed: List[str] = Field(description="Agents required")
    expected_output: str = Field(description="What the final output should contain")


class KPIItem(BaseModel):
    name: str
    value: str


class ExecutiveSummary(BaseModel):
    summary: str = Field(description="2-3 paragraph executive summary")
    key_insights: List[str] = Field(description="3-5 key insights")
    recommendations: List[str] = Field(description="Top 3 recommended actions")
    kpis: List[KPIItem] = Field(description="Key performance indicators as list")


# ── Supervisor state ──────────────────────────────────────────────────────────

class SupervisorState(TypedDict):
    session_id: str
    user_query: str
    context: Dict[str, Any]
    task_plan: Optional[TaskPlan]
    selected_agents: List[str]
    agent_results: Dict[str, AgentResult]
    final_report: Optional[str]
    executive_summary: Optional[ExecutiveSummary]
    error: Optional[str]


# ── Supervisor Agent ──────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Supervisor agent that orchestrates multiple specialized agents.

    Model routing:
      - Supervisor LLM  → GEMINI_SUPERVISOR_MODEL (default: gemini-2.5-flash)
      - Worker agents   → GEMINI_WORKER_MODEL     (default: gemini-2.5-flash-lite)

    Rate-limit protection:
      - Sequential execution with AGENT_CALL_DELAY_MS between calls
      - gemini_retry() on every LLM invocation (3 retries, exponential backoff)
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        llm: Optional[ChatGoogleGenerativeAI] = None,
    ):
        self.registry = agent_registry

        supervisor_model = _supervisor_model()
        self.llm = llm or ChatGoogleGenerativeAI(
            model=supervisor_model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )

        # Structured-output variants
        self.llm_router = self.llm.with_structured_output(AgentSelection)
        self.llm_planner = self.llm.with_structured_output(TaskPlan)
        self.llm_summarizer = self.llm.with_structured_output(ExecutiveSummary)

        self.graph = self._build_graph()

        logger.info(
            f"✅ Supervisor initialized | model={supervisor_model} | "
            f"agents={len(self.registry)} | delay={_agent_call_delay():.1f}s"
        )

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SupervisorState)
        workflow.add_node("parse_query",       self.parse_query_node)
        workflow.add_node("plan_task",         self.plan_task_node)
        workflow.add_node("select_agents",     self.select_agents_node)
        workflow.add_node("execute_agents",    self.execute_agents_node)
        workflow.add_node("aggregate_results", self.aggregate_results_node)
        workflow.add_node("generate_report",   self.generate_report_node)

        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query",       "plan_task")
        workflow.add_edge("plan_task",         "select_agents")
        workflow.add_edge("select_agents",     "execute_agents")
        workflow.add_edge("execute_agents",    "aggregate_results")
        workflow.add_edge("aggregate_results", "generate_report")
        workflow.add_edge("generate_report",   END)
        return workflow.compile()

    # ── Node implementations ──────────────────────────────────────────────────

    def parse_query_node(self, state: SupervisorState) -> SupervisorState:
        logger.info(f"📋 Parsing query: {state['user_query'][:100]}")
        state['context']['timestamp'] = datetime.now().isoformat()
        state['context']['available_agents'] = self.registry.list_agents()
        return state

    def plan_task_node(self, state: SupervisorState) -> SupervisorState:
        logger.info("📝 Planning task...")
        prompt = (
            "You are a task planning AI for a supply chain intelligence platform.\n\n"
            f"Available agents:\n{self._format_agent_capabilities()}\n\n"
            f"User query: {state['user_query']}\n\n"
            "Create a step-by-step plan. Determine which agents are needed and "
            "what the final output should contain."
        )
        try:
            task_plan: TaskPlan = gemini_retry(
                lambda: self.llm_planner.invoke(prompt)
            )
            state['task_plan'] = task_plan
            logger.info(f"  Plan: {len(task_plan.steps)} steps")
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            state['error'] = f"Planning failed: {e}"
        return state

    def select_agents_node(self, state: SupervisorState) -> SupervisorState:
        logger.info("🎯 Selecting agents...")
        prompt = (
            "You are an agent router for a supply chain intelligence platform.\n\n"
            f"Available agents:\n{self._format_agent_capabilities()}\n\n"
            f"User query: {state['user_query']}\n\n"
            f"Task plan: {state['task_plan'].steps if state.get('task_plan') else 'None'}\n\n"
            "Select the minimal set of agents needed. Return agent names, reasoning, "
            "and execution_order ('parallel' or 'sequential')."
        )
        try:
            selection: AgentSelection = gemini_retry(
                lambda: self.llm_router.invoke(prompt)
            )
            valid_agents = [
                a for a in selection.agents if a in self.registry.list_agents()
            ]
            if not valid_agents:
                best = self.registry.find_best_agent(state['user_query'])
                valid_agents = [best.name] if best else []

            state['selected_agents'] = valid_agents
            logger.info(f"  Selected: {valid_agents} | order: {selection.execution_order}")
            state['context']['execution_order'] = selection.execution_order
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            state['error'] = f"Agent selection failed: {e}"
        return state

    def execute_agents_node(self, state: SupervisorState) -> SupervisorState:
        """Execute agents sequentially with inter-call throttle."""
        logger.info("🚀 Executing agents (sequential + throttle)...")
        if not state['selected_agents']:
            logger.warning("No agents to execute")
            return state
        try:
            results = self._execute_sequential_throttled(
                state['selected_agents'],
                state['user_query'],
                state['context'],
            )
            state['agent_results'] = results
            for name, r in results.items():
                icon = "✅" if r.success else "❌"
                logger.info(f"  {icon} {name}: {len(r.insights)} insights")
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            state['error'] = f"Agent execution failed: {e}"
        return state

    def _execute_sequential_throttled(
        self,
        agent_names: List[str],
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, AgentResult]:
        """
        Run agents one-by-one with AGENT_CALL_DELAY_MS between each call.
        Each call is wrapped in gemini_retry for automatic 429 recovery.
        """
        delay = _agent_call_delay()
        logger.info(
            f"  Sequential throttle: {len(agent_names)} agents, "
            f"{delay:.1f}s delay between calls"
        )
        results: Dict[str, AgentResult] = {}
        accumulated_context = context.copy()

        for i, agent_name in enumerate(agent_names):
            agent = self.registry.get_agent(agent_name)
            if not agent:
                logger.warning(f"Agent '{agent_name}' not found — skipping")
                continue

            # Apply delay between calls (not before the first one)
            if i > 0:
                logger.info(f"  ⏱  Waiting {delay:.1f}s before '{agent_name}'...")
                time.sleep(delay)

            try:
                result = gemini_retry(
                    lambda a=agent: a.execute(query, accumulated_context),
                    max_retries=3,
                    base_delay=5,
                )
                results[agent_name] = result
                accumulated_context[f"{agent_name}_result"] = result
            except Exception as e:
                logger.error(f"Agent '{agent_name}' failed after retries: {e}")
                results[agent_name] = AgentResult(
                    agent_name=agent_name,
                    query=query,
                    timestamp=datetime.now(),
                    success=False,
                    error=str(e),
                )

        return results

    def aggregate_results_node(self, state: SupervisorState) -> SupervisorState:
        logger.info("📊 Aggregating results...")
        if not state['agent_results']:
            return state

        all_insights: List[str] = []
        all_recommendations: List[str] = []
        all_metrics: Dict[str, Any] = {}

        for agent_name, result in state['agent_results'].items():
            if result.success:
                all_insights.extend(result.insights)
                all_recommendations.extend(result.recommendations)
                all_metrics[agent_name] = result.metrics

        logger.info(f"  Insights: {len(all_insights)} | Recommendations: {len(all_recommendations)}")
        state['context']['aggregated_insights'] = all_insights
        state['context']['aggregated_recommendations'] = all_recommendations
        state['context']['aggregated_metrics'] = all_metrics
        return state

    def generate_report_node(self, state: SupervisorState) -> SupervisorState:
        logger.info("📄 Generating executive report...")
        if not state['agent_results']:
            state['final_report'] = "No results available to generate report."
            return state

        results_summary = self._format_results_for_llm(state['agent_results'])
        prompt = (
            "You are an executive report writer for OmniSupply platform.\n\n"
            f"User query: {state['user_query']}\n\n"
            f"Agent results:\n{results_summary}\n\n"
            "Create an executive summary with: summary (2-3 paragraphs), "
            "key_insights (3-5 bullets), recommendations (top 3 actions), "
            "kpis (key metrics). Be clear, concise, and actionable."
        )
        try:
            exec_summary: ExecutiveSummary = gemini_retry(
                lambda: self.llm_summarizer.invoke(prompt),
                max_retries=3,
                base_delay=5,
            )
            state['executive_summary'] = exec_summary
            state['final_report'] = self._build_report(state, exec_summary)
            logger.info("  ✅ Report generated")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state['error'] = f"Report generation failed: {e}"
            state['final_report'] = f"Error generating report: {e}"
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_agent_capabilities(self) -> str:
        lines = []
        for name in self.registry.list_agents():
            agent = self.registry.get_agent(name)
            if agent:
                lines.append(f"- {name}: {', '.join(agent.get_capabilities())}")
        return "\n".join(lines)

    def _format_results_for_llm(self, results: Dict[str, AgentResult]) -> str:
        lines = []
        for name, result in results.items():
            lines.append(f"\n**{name}**:")
            if result.success:
                for insight in result.insights[:3]:
                    lines.append(f"  - {insight}")
                for rec in result.recommendations[:2]:
                    lines.append(f"  → {rec}")
            else:
                lines.append(f"  Error: {result.error}")
        return "\n".join(lines)

    def _build_report(self, state: SupervisorState, summary: ExecutiveSummary) -> str:
        report = (
            f"# OmniSupply Intelligence Report\n\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Query:** {state['user_query']}\n"
            f"**Agents Used:** {', '.join(state['selected_agents'])}\n\n"
            f"---\n\n## Executive Summary\n\n{summary.summary}\n\n"
            f"---\n\n## Key Insights\n\n"
        )
        for i, insight in enumerate(summary.key_insights, 1):
            report += f"{i}. {insight}\n"

        report += "\n---\n\n## Recommended Actions\n\n"
        for i, rec in enumerate(summary.recommendations, 1):
            report += f"{i}. {rec}\n"

        report += "\n---\n\n## Key Performance Indicators\n\n"
        for kpi in summary.kpis:
            report += f"- **{kpi.name}**: {kpi.value}\n"

        report += "\n---\n\n## Detailed Results by Agent\n\n"
        for name, result in state['agent_results'].items():
            report += f"### {name}\n\n"
            if result.success:
                if result.insights:
                    report += "**Insights:**\n"
                    for insight in result.insights:
                        report += f"- {insight}\n"
                if result.metrics:
                    report += "\n**Metrics:**\n"
                    for k, v in result.metrics.items():
                        report += f"- {k}: {v}\n"
            else:
                report += f"*Error: {result.error}*\n"
            report += "\n"

        return report

    # ── Public execute ────────────────────────────────────────────────────────

    @track(project_name=OPIK_PROJECT_NAME)
    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full supervisor workflow for a user query."""
        logger.info(f"🎯 Supervisor executing: {query}")
        initial_state: SupervisorState = {
            "session_id": f"supervisor_{datetime.now().timestamp()}",
            "user_query": query,
            "context": context or {},
            "task_plan": None,
            "selected_agents": [],
            "agent_results": {},
            "final_report": None,
            "executive_summary": None,
            "error": None,
        }
        result = self.graph.invoke(initial_state, config={"callbacks": _get_callbacks()})
        logger.info("✅ Supervisor execution complete")
        return result
