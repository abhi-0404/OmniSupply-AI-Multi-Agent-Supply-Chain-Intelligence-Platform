# Tasks: Model Routing & Rate Limiting

## Task 1 — Create `src/utils/retry.py`
- [ ] Create `src/utils/__init__.py`
- [ ] Implement `_is_rate_limit(e)` helper
- [ ] Implement `gemini_retry(fn, max_retries, base_delay)` with exponential backoff
- [ ] Add logging for each retry attempt

## Task 2 — Update `config/settings.py`
- [ ] Add `GEMINI_SUPERVISOR_MODEL` field (default: `gemini-2.5-flash`)
- [ ] Add `GEMINI_WORKER_MODEL` field (default: `gemini-2.5-flash-lite`)
- [ ] Add `AGENT_CALL_DELAY_MS` field (default: `2000`)

## Task 3 — Update `src/agents/base.py`
- [ ] Change default model to read `GEMINI_WORKER_MODEL` (fallback to `GEMINI_MODEL`)
- [ ] Import and apply `gemini_retry` around `self.graph.invoke()`

## Task 4 — Update `src/supervisor/orchestrator.py`
- [ ] Change default model to read `GEMINI_SUPERVISOR_MODEL` (fallback to `GEMINI_MODEL`)
- [ ] Add `AGENT_CALL_DELAY_MS` env read
- [ ] Rename `_execute_parallel` → `_execute_sequential_throttled` with inter-call sleep
- [ ] Update `execute_agents_node` to always call `_execute_sequential_throttled`
- [ ] Wrap `self.llm_summarizer.invoke()` in `generate_report_node` with `gemini_retry`

## Task 5 — Update `.env.example`
- [ ] Add `GEMINI_SUPERVISOR_MODEL` with comment
- [ ] Add `GEMINI_WORKER_MODEL` with comment
- [ ] Add `AGENT_CALL_DELAY_MS` with comment

## Task 6 — Update `verify_setup.py`
- [ ] Check `GEMINI_WORKER_MODEL` or `GEMINI_MODEL` is set
- [ ] Print active supervisor and worker model names
