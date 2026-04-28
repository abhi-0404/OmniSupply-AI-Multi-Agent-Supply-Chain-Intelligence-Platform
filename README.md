# OmniSupply: Multi-Agent Supply Chain Intelligence Platform

**Enterprise AI system powered by Google Gemini for automated supply chain insights, risk predictions, and executive reporting.**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## What is OmniSupply?

OmniSupply is a production-ready multi-agent AI platform that ingests real supply chain, sales, and financial data to provide:

- **Automated Insights** — AI-generated KPI summaries, trend analysis, anomaly detection
- **Risk Predictions** — Proactive alerts for delivery delays, inventory shortages, quality issues
- **Process Optimization** — Data-driven recommendations for cost reduction and efficiency
- **Executive Reporting** — Weekly/monthly CxO-level business intelligence reports
- **Workflow Automation** — Stakeholder alerts, task creation, meeting agendas

---

## Architecture

```
User Query
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
| Retry / Rate Limiting | `src/utils/retry.py` — exponential backoff on 429s |
| Observability | Opik (optional) |

---

## Quick Start

### 1. Clone

```bash
git clone --depth 1 https://github.com/Bhardwaj-Saurabh/OmniSupply-AI-Multi-Agent-Supply-Chain-Intelligence-Platform.git
cd OmniSupply-AI-Multi-Agent-Supply-Chain-Intelligence-Platform
```

### 2. Create virtual environment and install dependencies

```bash
# Create venv
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install langchain-google-genai google-generativeai

# macOS / Linux
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install langchain-google-genai google-generativeai
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required — get your free key at https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-google-gemini-api-key-here

# Model routing (supervisor uses more capable model, workers use higher-quota model)
GEMINI_SUPERVISOR_MODEL=gemini-2.5-flash
GEMINI_WORKER_MODEL=gemini-2.5-flash-lite

# Rate limiting — delay between agent calls in ms (increase if hitting 429s)
AGENT_CALL_DELAY_MS=2000

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=omnisupply
POSTGRES_PASSWORD=omnisupply123
POSTGRES_DB=omnisupply
```

### 4. Start PostgreSQL

```bash
docker run -d --name omnisupply-db \
  -e POSTGRES_USER=omnisupply \
  -e POSTGRES_PASSWORD=omnisupply123 \
  -e POSTGRES_DB=omnisupply \
  -p 5432:5432 \
  postgres:15
```

### 5. Verify setup

```bash
# Windows
.venv\Scripts\python verify_setup.py

# macOS / Linux
.venv/bin/python verify_setup.py
```

Expected output: **18/18 checks passed**

### 6. Run the demo

```bash
# Fastest — skip data loading, use empty DB (agents still work via LLM)
.venv\Scripts\python omnisupply_demo.py --skip-data

# Auto-generate sample data then run
.venv\Scripts\python omnisupply_demo.py

# Force reload all data from CSV files in data/
.venv\Scripts\python omnisupply_demo.py --reload
```

---

## Running & Testing

### What the demo does

```
STEP 1  Load Data
        ✅ Connect to PostgreSQL
        ✅ Generate sample data (or load from CSV)

STEP 2  Initialize Agents
        ✅ data_analyst  → gemini-2.5-flash-lite
        ✅ risk_agent    → gemini-2.5-flash-lite
        ✅ finance_agent → gemini-2.5-flash-lite
        ✅ meeting_agent → gemini-2.5-flash-lite
        ✅ email_agent   → gemini-2.5-flash-lite

STEP 3  Test Individual Agents
        1️⃣  Data Analyst  — SQL generation + trend analysis
        2️⃣  Risk Agent    — Multi-dimensional risk scoring
        3️⃣  Finance Agent — P&L report + cashflow forecast

STEP 4  Supervisor Orchestration
        → gemini-2.5-flash plans and routes the query
        → Workers run sequentially (2s delay between each)
        → Exponential backoff on any 429 errors (5s → 10s → 20s)
        → Executive report generated
```

### Test a single agent

```python
# test_agent.py
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from src.storage.sql.database import DatabaseClient
from src.storage.vector.chromadb_client import OmniSupplyVectorStore
from src.agents import RiskAgent

db = DatabaseClient(database_url=os.getenv("DATABASE_URL"))
vs = OmniSupplyVectorStore()

agent = RiskAgent(db_client=db, vector_store=vs)
result = agent.execute("What are the current supply chain risks?")

print(f"Success: {result.success}  |  Time: {result.execution_time_ms:.0f}ms")
for insight in result.insights:
    print(insight)
```

```bash
.venv\Scripts\python test_agent.py
```

### Test the supervisor

```python
# test_supervisor.py
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from src.storage.sql.database import DatabaseClient
from src.storage.vector.chromadb_client import OmniSupplyVectorStore
from src.agents import AgentRegistry, DataAnalystAgent, RiskAgent, FinanceAgent
from src.supervisor.orchestrator import SupervisorAgent

db = DatabaseClient(database_url=os.getenv("DATABASE_URL"))
vs = OmniSupplyVectorStore()

registry = AgentRegistry()
registry.register(DataAnalystAgent(db_client=db, vector_store=vs))
registry.register(RiskAgent(db_client=db, vector_store=vs))
registry.register(FinanceAgent(db_client=db, vector_store=vs))

supervisor = SupervisorAgent(agent_registry=registry)
result = supervisor.execute(
    "Generate a weekly executive report with KPIs and risk summary"
)
print(result['final_report'])
```

```bash
.venv\Scripts\python test_supervisor.py
```

### Useful commands

```bash
# Check database record counts
.venv\Scripts\python -c "
import sys; sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv()
from src.storage.sql.database import DatabaseClient
import os
db = DatabaseClient(database_url=os.getenv('DATABASE_URL'))
print(db.get_table_counts())
"

# Clear database and reload
.venv\Scripts\python omnisupply_demo.py --reload

# Docker management
docker logs omnisupply-db
docker stop omnisupply-db
docker start omnisupply-db
```

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
GEMINI_SUPERVISOR_MODEL=gemini-2.5-flash      # or gemini-2.0-flash for paid
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

## Agent Capabilities

### Data Analyst Agent
- Natural language → SQL query generation (PostgreSQL)
- Data aggregation, trend analysis, KPI calculation
- Anomaly detection and visualization recommendations
- Auto-retry on SQL errors (up to 2 attempts with error context)

### Supply Chain Risk Agent
- Multi-dimensional risk scoring: delivery, inventory, quality, financial
- Late delivery prediction and carrier performance analysis
- Inventory shortage and overstock alerts
- Weighted aggregate risk score with severity classification

### Finance Insight Agent
- P&L report generation (revenue, COGS, gross/net profit, margins)
- Expense analysis and anomaly detection by category
- 90-day cashflow forecasting with confidence level
- Revenue growth comparison vs prior period

### Meeting / Report Agent
- Weekly and monthly executive reports
- CxO-level summaries with top 3 recommended actions
- Cross-agent data aggregation from registry
- Markdown-formatted output

### Email / Workflow Agent
- Stakeholder alert generation (INFO / WARNING / CRITICAL)
- Task creation with priority, assignee, and due date
- Email notification drafting
- Meeting agenda preparation

---

## Project Structure

```
OmniSupply/
├── src/
│   ├── data/
│   │   ├── models.py                  # Pydantic data models
│   │   └── ingestion/
│   │       ├── loaders.py             # CSV → Pydantic loaders (with sample data fallback)
│   │       └── validators.py          # Data quality checks
│   ├── storage/
│   │   ├── sql/
│   │   │   ├── models.py              # SQLAlchemy ORM models
│   │   │   └── database.py            # PostgreSQL client
│   │   └── vector/
│   │       ├── embeddings.py          # Gemini text-embedding-004 service
│   │       └── chromadb_client.py     # ChromaDB vector store
│   ├── agents/
│   │   ├── base.py                    # BaseAgent + AgentRegistry + _worker_model()
│   │   ├── data_analyst.py
│   │   ├── risk_agent.py
│   │   ├── finance_agent.py
│   │   ├── meeting_agent.py
│   │   └── email_agent.py
│   ├── supervisor/
│   │   └── orchestrator.py            # SupervisorAgent — sequential throttle + retry
│   └── utils/
│       └── retry.py                   # gemini_retry() — exponential backoff on 429s
├── config/
│   └── settings.py                    # Pydantic settings (all env vars)
├── .kiro/specs/
│   └── model-routing-rate-limiting/   # Spec: requirements, design, tasks
├── notebooks/                         # Jupyter exploration notebooks
├── data/                              # CSV data files (gitignored)
├── omnisupply_demo.py                 # Main demo script
├── verify_setup.py                    # Setup health check (18 checks)
├── requirements.txt
├── .env.example                       # Template with all variables documented
└── .gitignore
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `GOOGLE_API_KEY not found` | Edit `.env`, set your key from [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `PostgreSQL connection refused` | Run the Docker command in Step 4 |
| `429 RESOURCE_EXHAUSTED` | Retry logic handles this automatically. If daily quota is exhausted, wait until midnight UTC or switch to `gemini-1.5-flash` |
| `gemini-2.5-flash-lite not found` | Set `GEMINI_WORKER_MODEL=gemini-2.0-flash-lite` in `.env` |
| `ModuleNotFoundError: langchain_google_genai` | Run `.venv\Scripts\pip install langchain-google-genai` |
| `No module named 'src.data'` | Run from the project root directory, not a subdirectory |
| `OSError: long path` (Windows) | Enable long paths: run `reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f` as Administrator |

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

### Phase 3 — API Layer 🔄 In Progress
- FastAPI REST endpoints
- Authentication and rate limiting
- OpenAPI documentation

### Phase 4 — Automation
- Celery task queue for scheduled reports
- Real-time monitoring dashboard
- Email / Slack integration

### Phase 5 — Deployment
- Docker Compose setup
- Kubernetes manifests
- CI/CD pipeline

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
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
- [Opik](https://www.comet.com/site/products/opik/) — LLM observability (optional)
