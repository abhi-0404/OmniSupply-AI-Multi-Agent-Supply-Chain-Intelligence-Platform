"""
OmniSupply Multi-Agent System Demo
Complete demonstration of the OmniSupply platform with all agents

Usage:
    python omnisupply_demo.py                    # Auto-detect and use existing data
    python omnisupply_demo.py --reload           # Clear and reload all data
    python omnisupply_demo.py --skip-data        # Skip data loading entirely
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure Google API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file with your Google Gemini API key.")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.ingestion.loaders import OmniSupplyDataLoader
from src.data.ingestion.validators import DataQualityChecker
from src.storage.sql.database import DatabaseClient
from src.storage.vector.chromadb_client import OmniSupplyVectorStore
from src.agents import (
    AgentRegistry,
    DataAnalystAgent,
    RiskAgent,
    FinanceAgent,
    MeetingAgent,
    EmailAgent
)
from src.supervisor.orchestrator import SupervisorAgent


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def run_with_retry(fn, label, max_retries=3, wait=10):
    """Run a function with retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                if attempt < max_retries - 1:
                    print(f"⚠️  Rate limit hit for {label}. Waiting {wait}s before retry ({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"❌ Rate limit exceeded for {label} after {max_retries} attempts.")
                    raise
            else:
                raise


def print_result(agent_name: str, result):
    """Print agent result in a formatted way"""
    print(f"\n📋 Results from {agent_name}:")
    print(f"{'─' * 80}")
    print(f"✅ Success: {result.success}")
    print(f"⏱️  Execution time: {result.execution_time_ms:.2f}ms" if result.execution_time_ms else "")

    if result.insights:
        print(f"\n💡 Insights:")
        for insight in result.insights[:5]:  # Show first 5
            print(f"   {insight}")

    if result.recommendations:
        print(f"\n🎯 Recommendations:")
        for rec in result.recommendations[:3]:  # Show first 3
            print(f"   • {rec}")

    if result.metrics:
        print(f"\n📊 Key Metrics:")
        for key, value in list(result.metrics.items())[:5]:  # Show first 5
            print(f"   • {key}: {value}")

    print(f"{'─' * 80}")


def main():
    """Main demo function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="OmniSupply Multi-Agent Platform Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Clear existing data and reload fresh from CSV files"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data loading entirely (use existing database)"
    )
    args = parser.parse_args()

    print_section("🚀 OmniSupply Multi-Agent Platform Demo")

    print("This demo will:")
    print("  1. ✅ Load and validate supply chain data")
    print("  2. ✅ Initialize all 5 specialized agents")
    print("  3. ✅ Test individual agent capabilities")
    print("  4. ✅ Demonstrate Supervisor multi-agent orchestration")

    # ========================================
    # STEP 1: Load Data
    # ========================================
    print_section("📥 STEP 1: Loading Data")

    data_dir = Path("data")

    # Handle command-line flags
    if args.skip_data:
        skip_data = True
        print("⏭️  Skipping data loading (--skip-data flag)")
    elif not data_dir.exists():
        print(f"⚠️  Warning: Data directory '{data_dir}' not found.")
        print("Creating sample data directory structure...")
        data_dir.mkdir(exist_ok=True)
        print("Please place your CSV files in the 'data/' directory:")
        print("  - retail_orders.csv")
        print("  - supply_chain.csv")
        print("  - inventory.csv")
        print("  - financial_data.csv")
        print("\nSkipping data loading for now...")
        skip_data = True
    else:
        skip_data = False

    # PostgreSQL-only connection (no fallback)
    import os
    os.makedirs("data", exist_ok=True)

    # Build PostgreSQL connection string from .env variables
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")

    # Validate PostgreSQL configuration
    if not all([postgres_host, postgres_user, postgres_password, postgres_db]):
        print("❌ ERROR: PostgreSQL configuration incomplete in .env file")
        print("Required variables: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST")
        sys.exit(1)

    database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    print(f"📊 Connecting to PostgreSQL: {postgres_host}:{postgres_port}/{postgres_db}")

    db = DatabaseClient(database_url=database_url)
    print(f"✅ Connected to PostgreSQL successfully!")

    vector_store = OmniSupplyVectorStore()

    if not skip_data:
        try:
            # Check if database already has data
            if db.has_data():
                existing_counts = db.get_table_counts()
                total_records = sum(existing_counts.values())
                print(f"✅ Database already contains {total_records:,} records:")
                for table, count in existing_counts.items():
                    print(f"   • {table}: {count:,} records")

                # Handle based on --reload flag
                if args.reload:
                    print("\n🗑️  Clearing existing data (--reload flag)...")
                    db.clear_all_data()
                    clear_existing = False  # Already cleared
                else:
                    # Automatically use existing data (no prompt)
                    print("\n📊 Using existing data (no reload needed)")
                    print("💡 Tip: Use --reload flag to clear and reload all data")
                    clear_existing = False
                    skip_data = True  # Skip loading since we have data
            else:
                clear_existing = False

            if not skip_data:
                # Load datasets
                print("\n📥 Loading datasets from files...")
                loader = OmniSupplyDataLoader(data_dir=str(data_dir))
                data = loader.load_all()

                print(f"✅ Loaded from files:")
                print(f"   • Orders: {len(data.get('orders', []))} records")
                print(f"   • Shipments: {len(data.get('shipments', []))} records")
                print(f"   • Inventory: {len(data.get('inventory', []))} records")
                print(f"   • Transactions: {len(data.get('transactions', []))} records")

                # Validate data
                print("\n🔍 Validating data quality...")
                checker = DataQualityChecker()
                validation_results = checker.check_all(data)

                for dataset_name, result in validation_results.items():
                    status_icon = "✅" if result.status == "PASSED" else "❌"
                    print(f"   {status_icon} {dataset_name}: {result.status} ({result.issues_found} issues)")

                # Store in database
                print("\n💾 Storing data in SQL database...")
                counts = db.load_all_data(data, clear_existing=clear_existing)

                final_counts = db.get_table_counts()
                print(f"✅ Database now contains:")
                for table, count in final_counts.items():
                    new_records = counts.get(table, 0)
                    print(f"   • {table}: {count} records ({new_records} new)")

                # Index for vector search (sample)
                print("\n🔍 Indexing data for semantic search...")
                if data.get('orders'):
                    sample_orders = [o.model_dump() for o in data['orders'][:200]]
                    vector_store.index_orders(sample_orders)
                    print(f"✅ Indexed {len(sample_orders)} orders for semantic search")

        except Exception as e:
            import traceback
            print(f"⚠️  Warning: Could not load data: {e}")
            print(traceback.format_exc())
            print("Continuing with empty database...")

    # ========================================
    # STEP 2: Initialize Agents
    # ========================================
    print_section("🤖 STEP 2: Initializing Agents")

    # Create agent registry
    registry = AgentRegistry()

    # Initialize all agents
    print("Creating specialized agents...")

    data_analyst = DataAnalystAgent(db_client=db, vector_store=vector_store)
    registry.register(data_analyst)
    print(f"✅ {data_analyst.name}: {', '.join(data_analyst.get_capabilities()[:3])}")

    risk_agent = RiskAgent(db_client=db, vector_store=vector_store)
    registry.register(risk_agent)
    print(f"✅ {risk_agent.name}: {', '.join(risk_agent.get_capabilities()[:3])}")

    finance_agent = FinanceAgent(db_client=db, vector_store=vector_store)
    registry.register(finance_agent)
    print(f"✅ {finance_agent.name}: {', '.join(finance_agent.get_capabilities()[:3])}")

    meeting_agent = MeetingAgent(db_client=db, agent_registry=registry, vector_store=vector_store)
    registry.register(meeting_agent)
    print(f"✅ {meeting_agent.name}: {', '.join(meeting_agent.get_capabilities()[:3])}")

    email_agent = EmailAgent(db_client=db, vector_store=vector_store)
    registry.register(email_agent)
    print(f"✅ {email_agent.name}: {', '.join(email_agent.get_capabilities()[:3])}")

    print(f"\n✅ Total agents registered: {len(registry.agents)}")

    # ========================================
    # STEP 3: Test Individual Agents
    # ========================================
    print_section("🧪 STEP 3: Testing Individual Agents")

    # Test Data Analyst Agent
    print("\n1️⃣ Testing Data Analyst Agent...")
    try:
        result = run_with_retry(
            lambda: data_analyst.execute("Show me the top 5 product categories by revenue"),
            "Data Analyst"
        )
        print_result("Data Analyst Agent", result)
    except Exception as e:
        print(f"⚠️  Data Analyst test failed: {e}")

    time.sleep(3)  # avoid rate limits between calls

    # Test Risk Agent
    print("\n2️⃣ Testing Risk Agent...")
    try:
        result = run_with_retry(
            lambda: risk_agent.execute("What are the current supply chain risks?"),
            "Risk Agent"
        )
        print_result("Risk Agent", result)
    except Exception as e:
        print(f"⚠️  Risk Agent test failed: {e}")

    time.sleep(3)

    # Test Finance Agent
    print("\n3️⃣ Testing Finance Agent...")
    try:
        result = run_with_retry(
            lambda: finance_agent.execute("Generate financial summary with P&L and KPIs"),
            "Finance Agent"
        )
        print_result("Finance Agent", result)
    except Exception as e:
        print(f"⚠️  Finance Agent test failed: {e}")

    # ========================================
    # STEP 4: Supervisor Orchestration
    # ========================================
    print_section("🎯 STEP 4: Supervisor Multi-Agent Orchestration")

    # Create supervisor
    print("Initializing Supervisor Agent...")
    supervisor = SupervisorAgent(agent_registry=registry)
    print("✅ Supervisor Agent ready\n")

    # Test complex multi-agent queries
    print("=" * 80)
    print("Testing Complex Query 1: Executive Weekly Report")
    print("=" * 80)

    try:
        query = "Generate a weekly executive report with top risks, financial KPIs, and recommended actions"
        print(f"\n📝 Query: {query}\n")
        time.sleep(5)  # brief pause before supervisor calls
        result = run_with_retry(
            lambda: supervisor.execute(query),
            "Supervisor Query 1",
            max_retries=3,
            wait=15
        )

        print("\n📊 Supervisor Orchestration Results:")
        print(f"{'─' * 80}")
        print(f"✅ Agents Invoked: {', '.join(result.get('agents_executed', []))}")
        print(f"⏱️  Total Execution Time: {result.get('total_execution_time', 'N/A')}")

        if result.get('final_report'):
            print(f"\n📄 Executive Report:")
            print(result['final_report'])

        print(f"{'─' * 80}")

    except Exception as e:
        print(f"⚠️  Supervisor test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Testing Complex Query 2: Risk Assessment & Alerts")
    print("=" * 80)

    try:
        query = "Identify critical supply chain risks and create alerts for stakeholders"
        print(f"\n📝 Query: {query}\n")
        time.sleep(5)
        result = run_with_retry(
            lambda: supervisor.execute(query),
            "Supervisor Query 2",
            max_retries=3,
            wait=15
        )

        print("\n📊 Supervisor Orchestration Results:")
        print(f"{'─' * 80}")
        print(f"✅ Agents Invoked: {', '.join(result.get('agents_executed', []))}")
        print(f"⏱️  Total Execution Time: {result.get('total_execution_time', 'N/A')}")

        if result.get('final_report'):
            print(f"\n📄 Executive Summary:")
            print(result['final_report'])

        print(f"{'─' * 80}")

    except Exception as e:
        print(f"⚠️  Supervisor test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # Summary
    # ========================================
    print_section("✅ Demo Complete!")

    print("What was demonstrated:")
    print("  ✅ Data ingestion and validation")
    print("  ✅ SQL database storage (PostgreSQL)")
    print("  ✅ Vector database indexing (ChromaDB)")
    print("  ✅ 5 specialized agents:")
    print("     • Data Analyst Agent")
    print("     • Supply Chain Risk Agent")
    print("     • Finance Insight Agent")
    print("     • Meeting/Report Agent")
    print("     • Email/Workflow Agent")
    print("  ✅ Supervisor Agent orchestration")
    print("  ✅ Multi-agent query routing")
    print("  ✅ Parallel agent execution")
    print("  ✅ Executive report generation")

    print("\n🎯 Next Steps:")
    print("  1. Review agent outputs and refine prompts")
    print("  2. Add more data sources")
    print("  3. Implement email sending (SMTP integration)")
    print("  4. Deploy as FastAPI service")
    print("  5. Add scheduled reports (Celery)")
    print("  6. Create monitoring dashboards")

    print("\n📚 Documentation:")
    print("  • README.md - Project overview")
    print("  • OMNISUPPLY_ARCHITECTURE.md - Technical details")
    print("  • QUICKSTART.md - Setup guide")
    print("  • IMPLEMENTATION_SUMMARY.md - What was built")

    print("\n🚀 Happy building with OmniSupply! 🎉\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
