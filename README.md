# OmniSupply: Multi-Agent Supply Chain Intelligence Platform

**Enterprise AI system powered by Google Gemini for automated supply chain insights, risk predictions, and executive reporting.**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.4-green.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## What is OmniSupply?

OmniSupply is a production-ready multi-agent AI platform that ingests real supply chain, sales, and financial data to provide:

- **Automated Insights** — AI-generated KPI summaries, trend analysis, anomaly detection
- **Risk Predictions** — Proactive alerts for delivery delays, inventory shortages, quality issues
- **Process Optimization** — Data-driven recommendations for cost reduction and efficiency
- **Executive Reporting** — Weekly/monthly CxO-level business intelligence reports
- **Workflow Automation** — Stakeholder alerts, task creation, meeting agendas
- **Web UI** — Clean Streamlit dashboard to interact with all agents in a browser

---

## Architecture

```
User Query (Web UI or CLI)
    │
    ▼
Supervisor Agent  (gemini-2.5-flash — planning & reporting)
    │
    │  Sequential execution with 2s throttle + exponential backoff retry
    │
    ├── Data Analyst Agent   (gemini-2.5-flash-lite)
    │     SQL generation, trend analysis, anomaly detection
    │
    ├── Risk Agent           (gemini-2.5-flash-lite)
    │     Delivery, inventory, quality, financial risk scoring
    │
    ├── Finance Agent        (gemini-2.5-flash-lite)
    │     P&L reports, expense analysis, cashflow forecasting
    │
    ├── Meeting Agent        (gemini-2.5-flash-lite)
    │     Executive summaries, weekly/monthly reports
    │
    └── Email Agent          (gemini-2.5-flash-lite)
          Stakeholder alerts, task creation, meeting agendas
    │
    ▼
Aggregation & Executive Report
    │
    ▼
PostgreSQL (structured data) + ChromaDB (semantic search)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Supervisor LLM | `gemini-2.5-flash` (planning + report generation) |
| Worker LLMs | `gemini-2.5-flash-lite` (all 5 agents) |
| Embeddings | `text-embedding-004` via `langchain-google-genai` |
| Agent Framework | LangGraph + LangChain |
| SQL Database | PostgreSQL 15 |
| Vector Database | ChromaDB |
| Web UI | Streamlit (runs on port 8502) |
| Package Manager | uv |
| Retry / Rate Limiting | `src/utils/retry.py` — exponential backoff on 429s |
| Observability | Opik (optional) |

---

## How to Run This Project

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- Docker (for PostgreSQL)
- A free Google Gemini API key from [aistudio.google.com](https://aistudio.google.com/app/apikey)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/abhi-0404/OmniSupply-AI-Multi-Agent-Supply-Chain-Intelligence-Platform.git
cd OmniSupply-AI-Multi-Agent-Supply-Chain-Intelligence-Platform
```

---

### Step 2 — Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### Step 3 — Create virtual environment and install dependencies

```bash
uv venv
uv pip install -r requirements.txt
uv pip install langchain-google-genai==2.0.10 google-generativeai streamlit
```

> **Note:** `langchain-google-genai` must be pinned to `2.0.10` for compatibility with `langchain-core 1.1.0`.

---

### Step 4 — Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# Required — get your free key at https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-google-gemini-api-key-here

# Model routing
GEMINI_SUPERVISOR_MODEL=gemini-2.5-flash
GEMINI_WORKER_MODEL=gemini-2.5-flash-lite

# Rate limiting (increase if hitting 429 errors)
AGENT_CALL_DELAY_MS=2000

# PostgreSQL connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=omnisupply
POSTGRES_PASSWORD=omnisupply123
POSTGRES_DB=omnisupply
DATABASE_URL=postgresql://omnisupply:omnisupply123@localhost:5432/omnisupply
```

---

### Step 5 — Start PostgreSQL with Docker

```bash
docker run -d --name omnisupply-db \
  -e POSTGRES_USER=omnisupply \
  -e POSTGRES_PASSWORD=omnisupply123 \
  -e POSTGRES_DB=omnisupply \
  -p 5432:5432 \
  postgres:15
```

Verify it's running:

```bash
docker ps
```

---

### Step 6 — Verify the setup

```bash
# Windows
.venv\Scripts\python verify_setup.py

# macOS / Linux
.venv/bin/python verify_setup.py
```

Expected output: **18/18 checks passed**

---

### Step 7 — Run the Web UI ⭐ (Recommended)

```bash
.venv\Scripts\streamlit.exe run app.py        # Windows
.venv/bin/streamlit run app.py                # macOS / Linux
```

Opens at **http://localhost:8502**

**How to use the UI:**
1. Enter your PostgreSQL URL in the sidebar (pre-filled from `.env`)
2. Click **Connect & Initialize** — loads all 5 agents
3. Choose mode: **Supervisor (Multi-Agent)** or **Single Agent**
4. Click a preset query button or type your own question
5. Click **▶ Run Query** and view results

---

### Step 8 — (Optional) Run the CLI demo instead

```bash
# Skip data loading — fastest start
.venv\Scripts\python omnisupply_demo.py --skip-data

# Auto-generate sample data then run
.venv\Scripts\python omnisupply_demo.py

# Force reload all data from CSV files in data/
.venv\Scripts\python omnisupply_demo.py --reload
```

---

## Project Structure

```
OmniSupply/
├── app.py                             # Streamlit web UI (run this)
├── omnisupply_demo.py                 # CLI demo script
├── verify_setup.py                    # Setup health check (18 checks)
├── .env.example                       # Environment variable template
├── .streamlit/
│   └── config.toml                    # Streamlit config (port 8502)
├── src/
│   ├── agents/
│   │   ├── base.py                    # BaseAgent + AgentRegistry
│   │   ├── data_analyst.py            # SQL generation, trend analysis
│   │   ├── risk_agent.py              # Multi-dimensional risk scoring
│   │   ├── finance_agent.py           # P&L, cashflow forecasting
│   │   ├── meeting_agent.py           # Executive reports
│   │   └── email_agent.py             # Stakeholder alerts
│   ├── supervisor/
│   │   └── orchestrator.py            # SupervisorAgent — routes & aggregates
│   ├── storage/
│   │   ├── sql/
│   │   │   ├── models.py              # SQLAlchemy ORM models
│   │   │   └── database.py            # PostgreSQL client
│   │   └── vector/
│   │       ├── embeddings.py          # Gemini embedding service
│   │       └── chromadb_client.py     # ChromaDB vector store
│   ├── data/
│   │   ├── models.py                  # Pydantic data models
│   │   └── ingestion/
│   │       ├── loaders.py             # CSV → Pydantic loaders
│   │       └── validators.py          # Data quality checks
│   └── utils/
│       └── retry.py                   # gemini_retry() — exponential backoff
├── config/
│   └── settings.py                    # Pydantic settings (all env vars)
├── notebooks/                         # Jupyter exploration notebooks
├── data/                              # CSV data files (gitignored)
├── requirements.txt
└── pyproject.toml
```

---

## Agent Capabilities

### 📊 Data Analyst Agent
- Natural language → SQL query generation (PostgreSQL)
- Data aggregation, trend analysis, KPI calculation
- Anomaly detection and visualization recommendations
- Auto-retry on SQL errors (up to 2 attempts with error context)

### ⚠️ Supply Chain Risk Agent
- Multi-dimensional risk scoring: delivery, inventory, quality, financial
- Late delivery prediction and carrier performance analysis
- Inventory shortage and overstock alerts
- Weighted aggregate risk score with severity classification

### 💰 Finance Insight Agent
- P&L report generation (revenue, COGS, gross/net profit, margins)
- Expense analysis and anomaly detection by category
- 90-day cashflow forecasting with confidence level
- Revenue growth comparison vs prior period

### 📋 Meeting / Report Agent
- Weekly and monthly executive reports
- CxO-level summaries with top 3 recommended actions
- Cross-agent data aggregation from registry
- Markdown-formatted output

### ✉️ Email / Workflow Agent
- Stakeholder alert generation (INFO / WARNING / CRITICAL)
- Task creation with priority, assignee, and due date
- Email notification drafting
- Meeting agenda preparation

---

## Model Routing & Rate Limiting

OmniSupply uses a split-model strategy to stay within free-tier quotas:

| Role | Model | Free Tier |
|------|-------|-----------|
| Supervisor (planning + reports) | `gemini-2.5-flash` | 20 req/day |
| All 5 worker agents | `gemini-2.5-flash-lite` | 1500 req/day |

**Sequential throttle** — agents run one at a time with `AGENT_CALL_DELAY_MS` (default 2000ms) between calls, preventing per-minute quota bursts.

**Exponential backoff** — any 429 / `RESOURCE_EXHAUSTED` error is automatically retried:

```
Attempt 1 → wait  5s
Attempt 2 → wait 10s
Attempt 3 → wait 20s
Give up   → raise exception
```

To tune for your quota, edit `.env`:

```env
GEMINI_SUPERVISOR_MODEL=gemini-2.5-flash      # or gemini-2.0-flash for paid tier
GEMINI_WORKER_MODEL=gemini-2.5-flash-lite     # or gemini-2.0-flash-lite
AGENT_CALL_DELAY_MS=3000                      # increase if still hitting 429s
```

---

## Data Files (Optional)

Place CSV files in `data/`. If absent, the system auto-generates sample data:

```
data/
├── retail_orders.csv       # order_id, category, sale_price, profit, ...
├── supply_chain.csv        # carrier, status, delivery dates, ...
├── inventory.csv           # sku, stock_quantity, reorder_level, ...
└── financial_data.csv      # type, amount, category, ...
```

Source: [DataCo SMART SUPPLY CHAIN dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `GOOGLE_API_KEY not found` | Edit `.env`, set your key from [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `PostgreSQL connection refused` | Run the Docker command in Step 5 |
| `429 RESOURCE_EXHAUSTED` | Retry logic handles automatically. If daily quota exhausted, wait until midnight UTC |
| `cannot import name 'ContextOverflowError'` | Run `uv pip install "langchain-google-genai==2.0.10"` |
| `No module named 'langchain_google_genai'` | Run `uv pip install langchain-google-genai==2.0.10 google-generativeai` |
| `streamlit not recognized` | Use `.venv\Scripts\streamlit.exe run app.py` instead of `streamlit run app.py` |
| `Connection failed: Access denied` | Stop any running Python processes, then retry the install |
| `No module named 'src.data'` | Run commands from the project root directory |
| `OSError: long path` (Windows) | Run as Administrator: `reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f` |
| `gemini-2.5-flash-lite not found` | Set `GEMINI_WORKER_MODEL=gemini-2.0-flash-lite` in `.env` |

---

## Roadmap

### Phase 1 — Core Platform ✅ Complete
- Data ingestion pipeline with validation and sample data fallback
- PostgreSQL + ChromaDB storage layer
- BaseAgent abstraction with LangGraph workflows
- 5 specialized agents

### Phase 2 — Gemini Integration ✅ Complete
- Migrated from OpenAI to Google Gemini
- Split model routing (supervisor vs workers)
- Sequential throttle with configurable delay
- Exponential backoff retry on 429 errors
- Gemini `text-embedding-004` for semantic search

### Phase 3 — Web UI ✅ Complete
- Streamlit dashboard on port 8502
- Supervisor and single-agent modes
- Preset query buttons, KPI tiles, per-agent result cards
- Query history in sidebar

### Phase 4 — API Layer 🔄 In Progress
- FastAPI REST endpoints
- Authentication and rate limiting
- OpenAPI documentation

### Phase 5 — Automation
- Celery task queue for scheduled reports
- Real-time monitoring dashboard
- Email / Slack integration

### Phase 6 — Deployment
- Docker Compose setup
- Kubernetes manifests
- CI/CD pipeline

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent workflows
- [Google Gemini](https://ai.google.dev/) — LLM and embeddings
- [ChromaDB](https://www.trychroma.com/) — Vector search
- [PostgreSQL](https://www.postgresql.org/) — Production database
- [Streamlit](https://streamlit.io/) — Web UI
- [uv](https://docs.astral.sh/uv/) — Fast Python package manager
- [Opik](https://www.comet.com/site/products/opik/) — LLM observability (optional)
