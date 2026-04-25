"""
Quick setup verification script for OmniSupply.
Run: .venv/Scripts/python verify_setup.py
"""

import sys
import os

def check(label, fn):
    try:
        fn()
        print(f"  ✅ {label}")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False

print("\n=== OmniSupply Setup Verification ===\n")

results = []

# Core deps
print("1. Core dependencies:")
results.append(check("langchain", lambda: __import__("langchain")))
results.append(check("langgraph", lambda: __import__("langgraph")))
results.append(check("langchain-google-genai", lambda: __import__("langchain_google_genai")))
results.append(check("chromadb", lambda: __import__("chromadb")))
results.append(check("sqlalchemy", lambda: __import__("sqlalchemy")))
results.append(check("psycopg2", lambda: __import__("psycopg2")))
results.append(check("opik", lambda: __import__("opik")))
results.append(check("pandas", lambda: __import__("pandas")))
results.append(check("pydantic", lambda: __import__("pydantic")))

# Project modules
print("\n2. Project modules:")
sys.path.insert(0, ".")
results.append(check("src.data.models", lambda: __import__("src.data.models", fromlist=["AgentResult"])))
results.append(check("src.data.ingestion.loaders", lambda: __import__("src.data.ingestion.loaders", fromlist=["OmniSupplyDataLoader"])))
results.append(check("src.data.ingestion.validators", lambda: __import__("src.data.ingestion.validators", fromlist=["DataQualityChecker"])))
results.append(check("src.storage.sql.database", lambda: __import__("src.storage.sql.database", fromlist=["DatabaseClient"])))
results.append(check("src.storage.vector.chromadb_client", lambda: __import__("src.storage.vector.chromadb_client", fromlist=["OmniSupplyVectorStore"])))
results.append(check("src.agents", lambda: __import__("src.agents", fromlist=["AgentRegistry"])))
results.append(check("src.supervisor.orchestrator", lambda: __import__("src.supervisor.orchestrator", fromlist=["SupervisorAgent"])))

# Environment variables
print("\n3. Environment variables:")
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key and api_key != "your-google-gemini-api-key-here":
    print("  ✅ GOOGLE_API_KEY is set")
    results.append(True)
else:
    print("  ⚠️  GOOGLE_API_KEY not set — edit .env file before running the demo")
    results.append(False)

# Model routing
supervisor_model = os.getenv("GEMINI_SUPERVISOR_MODEL") or os.getenv("GEMINI_MODEL", "")
worker_model = os.getenv("GEMINI_WORKER_MODEL") or os.getenv("GEMINI_MODEL", "")
delay_ms = os.getenv("AGENT_CALL_DELAY_MS", "2000")

if worker_model:
    print(f"  ✅ Worker model:     {worker_model}")
    results.append(True)
else:
    print("  ⚠️  No worker model set (GEMINI_WORKER_MODEL or GEMINI_MODEL)")
    results.append(False)

print(f"  ✅ Supervisor model: {supervisor_model or '(fallback to worker)'}")
print(f"  ✅ Agent call delay: {delay_ms}ms")

pg_host = os.getenv("POSTGRES_HOST", "")
if pg_host:
    print(f"  ✅ PostgreSQL configured: {pg_host}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', '')}")
else:
    print("  ⚠️  PostgreSQL not configured in .env")

print("\n" + "="*40)
passed = sum(results)
total = len(results)
print(f"Result: {passed}/{total} checks passed")

if passed >= total - 1:
    print("\n🚀 Setup looks good! Next steps:")
    print("   1. Edit .env and set your GOOGLE_API_KEY")
    print("   2. Start PostgreSQL (Docker: docker run -e POSTGRES_USER=omnisupply -e POSTGRES_PASSWORD=omnisupply123 -e POSTGRES_DB=omnisupply -p 5432:5432 -d postgres:15)")
    print("   3. Run: .venv/Scripts/python omnisupply_demo.py --skip-data")
else:
    print("\n⚠️  Some checks failed. Fix the issues above.")
