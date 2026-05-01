"""
OmniSupply AI — Streamlit Web Interface
Run with: streamlit run app.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="OmniSupply AI",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }

/* ── Typography ── */
h1, h2, h3 { color: #e2e8f0 !important; }
p, li, label { color: #94a3b8 !important; }

/* ── Cards ── */
.card {
    background: #1e2535;
    border: 1px solid #2a3347;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-accent {
    background: linear-gradient(135deg, #1e2535 0%, #1a2540 100%);
    border: 1px solid #3b4fd8;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.metric-tile {
    background: #1e2535;
    border: 1px solid #2a3347;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    min-width: 140px;
}
.metric-tile .label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.metric-tile .value { font-size: 22px; font-weight: 700; color: #e2e8f0; margin-top: 4px; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
}
.badge-blue  { background: #1e3a5f; color: #60a5fa; }
.badge-green { background: #14532d; color: #4ade80; }
.badge-amber { background: #451a03; color: #fbbf24; }
.badge-red   { background: #450a0a; color: #f87171; }

/* ── Insight / recommendation items ── */
.insight-item {
    background: #151c2c;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: #cbd5e1 !important;
    font-size: 14px;
}
.rec-item {
    background: #151c2c;
    border-left: 3px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: #cbd5e1 !important;
    font-size: 14px;
}

/* ── Agent status pills ── */
.agent-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e2535;
    border: 1px solid #2a3347;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #94a3b8;
    margin: 3px;
}
.agent-pill.active { border-color: #3b82f6; color: #60a5fa; background: #1e3a5f22; }

/* ── Report output ── */
.report-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 24px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    color: #d1d5db;
    max-height: 600px;
    overflow-y: auto;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: #1e2535 !important;
    border: 1px solid #2a3347 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3b4fd8, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    width: 100%;
}
.stButton > button:hover { opacity: .9; }
.stSelectbox > div > div {
    background: #1e2535 !important;
    border: 1px solid #2a3347 !important;
    color: #e2e8f0 !important;
}
div[data-testid="stExpander"] {
    background: #1e2535;
    border: 1px solid #2a3347;
    border-radius: 10px;
}
.stSpinner > div { color: #60a5fa !important; }
hr { border-color: #2a3347 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "db_client": None,
        "vector_store": None,
        "agent_registry": None,
        "supervisor": None,
        "connected": False,
        "history": [],          # list of {query, result, ts, mode}
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Backend helpers ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build_backend(database_url: str):
    """Initialise DB, vector store, agents, supervisor — cached per URL."""
    from src.storage.sql.database import DatabaseClient
    from src.storage.vector.chromadb_client import OmniSupplyVectorStore
    from src.agents import (
        AgentRegistry, DataAnalystAgent, RiskAgent,
        FinanceAgent, MeetingAgent, EmailAgent,
    )
    from src.supervisor.orchestrator import SupervisorAgent

    db = DatabaseClient(database_url=database_url)
    vs = OmniSupplyVectorStore()

    registry = AgentRegistry()
    registry.register(DataAnalystAgent(db_client=db, vector_store=vs))
    registry.register(RiskAgent(db_client=db, vector_store=vs))
    registry.register(FinanceAgent(db_client=db, vector_store=vs))
    registry.register(MeetingAgent(db_client=db, vector_store=vs))
    registry.register(EmailAgent(db_client=db, vector_store=vs))

    supervisor = SupervisorAgent(agent_registry=registry)
    return db, vs, registry, supervisor


AGENT_META = {
    "data_analyst": {"icon": "📊", "label": "Data Analyst",  "color": "badge-blue"},
    "risk_agent":   {"icon": "⚠️", "label": "Risk Agent",    "color": "badge-amber"},
    "finance_agent":{"icon": "💰", "label": "Finance Agent", "color": "badge-green"},
    "meeting_agent":{"icon": "📋", "label": "Meeting Agent", "color": "badge-blue"},
    "email_agent":  {"icon": "✉️", "label": "Email Agent",   "color": "badge-blue"},
}

PRESET_QUERIES = {
    "📊 Weekly KPI Summary":
        "Generate a weekly executive report with key KPIs, revenue trends, and top-performing categories.",
    "⚠️ Supply Chain Risk Scan":
        "Identify current supply chain risks including delivery delays, inventory shortages, and quality issues.",
    "💰 Financial Health Check":
        "Provide a P&L summary, expense breakdown, and 90-day cashflow forecast.",
    "📋 Executive Briefing":
        "Create a monthly executive briefing with cross-functional insights and top 3 recommended actions.",
    "✉️ Stakeholder Alert Draft":
        "Draft stakeholder alerts for any critical supply chain or financial issues that need immediate attention.",
    "🔍 Anomaly Detection":
        "Detect any anomalies in orders, shipments, inventory levels, or financial transactions.",
}


# ── Rendering helpers ──────────────────────────────────────────────────────────
def _severity_badge(score: float) -> str:
    if score >= 0.7:
        return '<span class="badge badge-red">HIGH</span>'
    if score >= 0.4:
        return '<span class="badge badge-amber">MEDIUM</span>'
    return '<span class="badge badge-green">LOW</span>'


def _render_agent_result(name: str, result):
    meta = AGENT_META.get(name, {"icon": "🤖", "label": name, "color": "badge-blue"})
    success_badge = (
        '<span class="badge badge-green">✓ Success</span>'
        if result.success else
        '<span class="badge badge-red">✗ Failed</span>'
    )
    exec_time = f"{result.execution_time_ms:.0f} ms" if result.execution_time_ms else "—"

    st.markdown(f"""
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
        <span style="font-size:16px; font-weight:700; color:#e2e8f0;">
          {meta['icon']} {meta['label']}
        </span>
        <span style="display:flex; gap:8px; align-items:center;">
          {success_badge}
          <span style="font-size:11px; color:#64748b;">⏱ {exec_time}</span>
        </span>
      </div>
    """, unsafe_allow_html=True)

    if not result.success and result.error:
        st.error(result.error)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)

    with col1:
        if result.insights:
            st.markdown("**💡 Insights**")
            for ins in result.insights[:6]:
                st.markdown(f'<div class="insight-item">{ins}</div>', unsafe_allow_html=True)

    with col2:
        if result.recommendations:
            st.markdown("**🎯 Recommendations**")
            for rec in result.recommendations[:4]:
                st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

    if result.metrics:
        st.markdown("**📈 Metrics**")
        cols = st.columns(min(len(result.metrics), 4))
        for i, (k, v) in enumerate(list(result.metrics.items())[:4]):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="metric-tile">
                  <div class="label">{k}</div>
                  <div class="value">{v}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_supervisor_result(state: dict):
    summary = state.get("executive_summary")
    report  = state.get("final_report", "")
    agents  = state.get("selected_agents", [])
    results = state.get("agent_results", {})

    # ── KPI strip ──
    if summary and summary.kpis:
        cols = st.columns(min(len(summary.kpis), 5))
        for i, kpi in enumerate(summary.kpis[:5]):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-tile">
                  <div class="label">{kpi.name}</div>
                  <div class="value">{kpi.value}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")

    # ── Agents used ──
    if agents:
        pills = "".join(
            f'<span class="agent-pill active">'
            f'{AGENT_META.get(a, {}).get("icon","🤖")} '
            f'{AGENT_META.get(a, {}).get("label", a)}'
            f'</span>'
            for a in agents
        )
        st.markdown(f"**Agents invoked:** {pills}", unsafe_allow_html=True)
        st.markdown("")

    # ── Executive summary ──
    if summary:
        st.markdown("### Executive Summary")
        st.markdown(f'<div class="card"><p style="color:#cbd5e1;line-height:1.8;">{summary.summary}</p></div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Insights**")
            for ins in summary.key_insights:
                st.markdown(f'<div class="insight-item">{ins}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Recommended Actions**")
            for rec in summary.recommendations:
                st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

    # ── Per-agent results ──
    if results:
        st.markdown("### Agent Results")
        for agent_name, result in results.items():
            _render_agent_result(agent_name, result)

    # ── Full report ──
    if report:
        with st.expander("📄 Full Report", expanded=False):
            st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)


def _render_single_result(result):
    _render_agent_result(result.agent_name, result)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 20px;">
          <div style="font-size:22px; font-weight:800; color:#e2e8f0;">🔗 OmniSupply AI</div>
          <div style="font-size:12px; color:#64748b; margin-top:2px;">Multi-Agent Supply Chain Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Connection ──
        st.markdown("#### Connection")
        db_url = st.text_input(
            "PostgreSQL URL",
            value=os.getenv("DATABASE_URL", "postgresql://omnisupply:omnisupply123@localhost:5432/omnisupply"),
            type="password",
            help="postgresql://user:pass@host:5432/dbname",
            label_visibility="collapsed",
            placeholder="postgresql://user:pass@host:5432/db",
        )

        if not st.session_state.connected:
            if st.button("Connect & Initialize", use_container_width=True):
                if not os.getenv("GOOGLE_API_KEY"):
                    st.error("GOOGLE_API_KEY not found in .env")
                else:
                    with st.spinner("Connecting…"):
                        try:
                            db, vs, registry, supervisor = _build_backend(db_url)
                            st.session_state.db_client      = db
                            st.session_state.vector_store   = vs
                            st.session_state.agent_registry = registry
                            st.session_state.supervisor     = supervisor
                            st.session_state.connected      = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Connection failed: {e}")
        else:
            st.markdown('<span class="badge badge-green">● Connected</span>', unsafe_allow_html=True)
            if st.button("Disconnect", use_container_width=True):
                for k in ["db_client","vector_store","agent_registry","supervisor","connected"]:
                    st.session_state[k] = None if k != "connected" else False
                st.rerun()

        st.divider()

        # ── Mode ──
        st.markdown("#### Mode")
        mode = st.radio(
            "mode",
            ["🧠 Supervisor (Multi-Agent)", "🎯 Single Agent"],
            label_visibility="collapsed",
        )

        single_agent = None
        if "Single Agent" in mode:
            single_agent = st.selectbox(
                "Agent",
                options=list(AGENT_META.keys()),
                format_func=lambda k: f"{AGENT_META[k]['icon']} {AGENT_META[k]['label']}",
            )

        st.divider()

        # ── History ──
        if st.session_state.history:
            st.markdown("#### History")
            for i, h in enumerate(reversed(st.session_state.history[-8:])):
                ts = h["ts"].strftime("%H:%M")
                label = h["query"][:34] + "…" if len(h["query"]) > 34 else h["query"]
                if st.button(f"{ts}  {label}", key=f"hist_{i}", use_container_width=True):
                    st.session_state.last_result = h
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_result = None
                st.rerun()

        return mode, single_agent


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    mode, single_agent = _sidebar()

    # ── Header ──
    st.markdown("""
    <div style="margin-bottom: 28px;">
      <h1 style="margin:0; font-size:28px;">Supply Chain Intelligence</h1>
      <p style="margin:4px 0 0; color:#64748b;">Ask anything about your supply chain, risk, finance, or operations.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.connected:
        st.markdown("""
        <div class="card-accent" style="text-align:center; padding: 40px;">
          <div style="font-size:40px; margin-bottom:12px;">🔗</div>
          <div style="font-size:18px; font-weight:700; color:#e2e8f0; margin-bottom:8px;">Connect to get started</div>
          <div style="color:#64748b;">Enter your PostgreSQL URL in the sidebar and click <strong>Connect & Initialize</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Preset queries ──
    st.markdown("**Quick queries**")
    preset_cols = st.columns(3)
    selected_preset = None
    for i, (label, query) in enumerate(PRESET_QUERIES.items()):
        with preset_cols[i % 3]:
            if st.button(label, key=f"preset_{i}", use_container_width=True):
                selected_preset = query

    st.markdown("")

    # ── Query input ──
    query_val = selected_preset or ""
    query = st.text_area(
        "Query",
        value=query_val,
        height=100,
        placeholder="e.g. What are the top supply chain risks this week? Generate a P&L summary. Draft a stakeholder alert for critical delays.",
        label_visibility="collapsed",
    )

    run_col, _ = st.columns([1, 3])
    with run_col:
        run = st.button("▶  Run Query", use_container_width=True)

    # ── Execute ──
    if run and query.strip():
        is_supervisor = "Supervisor" in mode

        with st.spinner("Running agents… this may take 20–60 seconds"):
            try:
                ts_start = time.time()

                if is_supervisor:
                    result_state = st.session_state.supervisor.execute(query)
                    elapsed = time.time() - ts_start
                    entry = {
                        "query": query,
                        "result": result_state,
                        "ts": datetime.now(),
                        "mode": "supervisor",
                        "elapsed": elapsed,
                    }
                else:
                    agent = st.session_state.agent_registry.get_agent(single_agent)
                    if agent is None:
                        st.error(f"Agent '{single_agent}' not found in registry.")
                        return
                    result = agent.execute(query)
                    elapsed = time.time() - ts_start
                    entry = {
                        "query": query,
                        "result": result,
                        "ts": datetime.now(),
                        "mode": "single",
                        "elapsed": elapsed,
                    }

                st.session_state.history.append(entry)
                st.session_state.last_result = entry

            except Exception as e:
                st.error(f"Execution failed: {e}")
                return

    # ── Results ──
    entry = st.session_state.last_result
    if entry:
        st.divider()

        elapsed_str = f"{entry['elapsed']:.1f}s" if "elapsed" in entry else ""
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
          <div style="font-size:13px; color:#64748b;">
            🕐 {entry['ts'].strftime('%b %d, %H:%M')}
            {"  ·  ⏱ " + elapsed_str if elapsed_str else ""}
          </div>
        </div>
        <div class="card" style="margin-bottom:20px;">
          <span style="font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.06em;">Query</span>
          <p style="color:#e2e8f0; margin:6px 0 0; font-size:15px;">{entry['query']}</p>
        </div>
        """, unsafe_allow_html=True)

        if entry["mode"] == "supervisor":
            _render_supervisor_result(entry["result"])
        else:
            _render_single_result(entry["result"])


if __name__ == "__main__":
    main()
