# Requirements: Model Routing & Rate Limiting

## Overview
OmniSupply hits Gemini free-tier quotas during multi-agent orchestration because all agents share one model and the supervisor fires them in parallel. This spec adds a model-routing layer, sequential throttling, exponential-backoff retry, and split environment variables so the platform runs reliably on a free-tier key.

---

## Requirements

### REQ-1 — Model Split (Supervisor vs Workers)
**User Story:** As a developer, I want the Supervisor to use a more capable model while worker agents use a lighter, higher-quota model, so I can balance quality and quota usage.

**Acceptance Criteria:**
- `GEMINI_SUPERVISOR_MODEL` env var controls the Supervisor LLM (default: `gemini-2.5-flash`)
- `GEMINI_WORKER_MODEL` env var controls all 5 worker agent LLMs (default: `gemini-2.5-flash-lite`)
- Legacy `GEMINI_MODEL` still works as a fallback for both if the split vars are absent
- Each agent reads its model from the correct env var at init time
- The split is documented in `.env.example`

### REQ-2 — Sequential Throttle in Supervisor
**User Story:** As a developer, I want agent calls inside the supervisor to run sequentially with a configurable delay, so I don't burst the per-minute quota.

**Acceptance Criteria:**
- `_execute_parallel` in `orchestrator.py` is replaced by a sequential loop
- A `AGENT_CALL_DELAY_MS` env var (default: `2000`) controls the inter-call delay in milliseconds
- The delay is applied **between** agent calls, not before the first one
- The execution order follows the `selected_agents` list
- Results are collected in the same dict format as before so downstream code is unchanged

### REQ-3 — Exponential Backoff Retry Wrapper
**User Story:** As a developer, I want any 429 / RESOURCE_EXHAUSTED error to be retried automatically with exponential backoff, so transient quota hits don't fail the whole run.

**Acceptance Criteria:**
- A reusable `gemini_retry(fn, max_retries=3, base_delay=5)` utility function lives in `src/utils/retry.py`
- Retry triggers on: HTTP 429, `RESOURCE_EXHAUSTED` in the error message
- Backoff formula: `base_delay * (2 ** attempt)` seconds (5s → 10s → 20s)
- After `max_retries` exhausted, the original exception is re-raised
- Each retry logs: attempt number, wait time, error snippet
- The wrapper is applied in:
  - `BaseAgent.execute()` around `self.graph.invoke()`
  - `SupervisorAgent._execute_sequential()` around each `agent.execute()` call
  - `SupervisorAgent.generate_report_node()` around the LLM summarizer call

### REQ-4 — Environment Variable Sync
**User Story:** As a developer, I want `.env.example` and `config/settings.py` to document all new variables, so teammates can onboard without guessing.

**Acceptance Criteria:**
- `.env.example` adds `GEMINI_SUPERVISOR_MODEL`, `GEMINI_WORKER_MODEL`, `AGENT_CALL_DELAY_MS`
- `config/settings.py` adds the three new fields with correct defaults and env mappings
- Old `GEMINI_MODEL` is kept as a deprecated fallback (not removed)
- `verify_setup.py` checks that at least one of `GEMINI_MODEL` or `GEMINI_WORKER_MODEL` is set
