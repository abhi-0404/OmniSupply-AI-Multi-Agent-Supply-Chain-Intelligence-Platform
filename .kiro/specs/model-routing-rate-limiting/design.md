# Design: Model Routing & Rate Limiting

## Architecture

```
.env
  GEMINI_SUPERVISOR_MODEL=gemini-2.5-flash
  GEMINI_WORKER_MODEL=gemini-2.5-flash-lite
  AGENT_CALL_DELAY_MS=2000
        │
        ▼
config/settings.py          ← adds 3 new fields
        │
        ├── src/utils/retry.py   ← NEW: gemini_retry() utility
        │
        ├── src/agents/base.py   ← reads GEMINI_WORKER_MODEL
        │                           wraps graph.invoke() with gemini_retry()
        │
        └── src/supervisor/orchestrator.py
                ├── reads GEMINI_SUPERVISOR_MODEL
                ├── _execute_sequential() replaces _execute_parallel()
                │     loop: agent.execute() → sleep(AGENT_CALL_DELAY_MS) → next
                └── generate_report_node() wrapped with gemini_retry()
```

## New File: `src/utils/retry.py`

```python
def gemini_retry(fn, max_retries=3, base_delay=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit (attempt {attempt+1}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
```

## Changes per File

| File | Change |
|------|--------|
| `src/utils/__init__.py` | New empty init |
| `src/utils/retry.py` | New — `gemini_retry()` + `_is_rate_limit()` |
| `src/agents/base.py` | Worker model env var; wrap `graph.invoke()` |
| `src/supervisor/orchestrator.py` | Supervisor model env var; sequential loop; wrap report LLM |
| `config/settings.py` | 3 new settings fields |
| `.env.example` | Document new vars |
| `verify_setup.py` | Check new vars |

## Sequential Execution Flow (replaces parallel)

```
execute_agents_node()
    │
    └── _execute_sequential_throttled(agent_names, query, context)
            for i, agent_name in enumerate(agent_names):
                result = gemini_retry(lambda: agent.execute(query, context))
                results[agent_name] = result
                if i < len(agent_names) - 1:
                    time.sleep(AGENT_CALL_DELAY_MS / 1000)
            return results
```

The supervisor's `execution_order` field from the LLM router is kept but `parallel` is now treated the same as `sequential` — the throttle always applies.

## Retry Backoff Table

| Attempt | Wait (base_delay=5s) |
|---------|---------------------|
| 1st retry | 5s |
| 2nd retry | 10s |
| 3rd retry | 20s |
| Give up | raise original exception |
