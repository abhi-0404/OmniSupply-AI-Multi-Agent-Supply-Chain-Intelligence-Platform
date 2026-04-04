"""
Data loaders for OmniSupply platform.
Loads CSV/Excel files into Pydantic models.
"""

import os
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from ..models import Order, Shipment, InventoryItem, FinancialTransaction

logger = logging.getLogger(__name__)


def _read_csv_safe(filepath: str) -> Optional[pd.DataFrame]:
    """Try reading a CSV with multiple encodings."""
    for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            logger.info(f"Read {filepath} with encoding {enc} ({len(df)} rows)")
            return df
        except Exception:
            continue
    logger.error(f"Could not read {filepath} with any encoding")
    return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(float(val)) if pd.notna(val) else None
    except Exception:
        return None


def _safe_bool(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "yes", "true", "y")


def _safe_date(val) -> Optional[datetime]:
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None


class OmniSupplyDataLoader:
    """Loads all OmniSupply datasets from a data directory."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_all(self) -> Dict[str, List[Any]]:
        """Load all available datasets."""
        result: Dict[str, List[Any]] = {}

        orders = self._load_orders()
        if orders:
            result["orders"] = orders

        shipments = self._load_shipments()
        if shipments:
            result["shipments"] = shipments

        inventory = self._load_inventory()
        if inventory:
            result["inventory"] = inventory

        transactions = self._load_transactions()
        if transactions:
            result["transactions"] = transactions

        return result

    # ── Orders ──────────────────────────────────────────────────────────────

    def _load_orders(self) -> List[Order]:
        """Load orders from retail_orders.csv (or similar)."""
        candidates = [
            "retail_orders.csv",
            "DataCoSupplyChainDataset.csv",
            "orders.csv",
        ]
        df = self._find_and_read(candidates)
        if df is None:
            logger.warning("No orders file found")
            return []

        orders: List[Order] = []
        col = {c.lower().strip(): c for c in df.columns}

        for _, row in df.iterrows():
            try:
                order_id = str(
                    row.get(col.get("order id", col.get("orderid", "")), uuid.uuid4())
                )
                raw_date = row.get(col.get("order date", col.get("orderdate", "")))
                order_date = _safe_date(raw_date) or datetime.now()

                orders.append(Order(
                    order_id=order_id,
                    order_date=order_date,
                    ship_mode=str(row.get(col.get("ship mode", col.get("shipmode", "")), "")) or None,
                    segment=str(row.get(col.get("segment", ""), "")) or None,
                    country=str(row.get(col.get("country", ""), "")) or None,
                    city=str(row.get(col.get("city", ""), "")) or None,
                    state=str(row.get(col.get("state", ""), "")) or None,
                    postal_code=str(row.get(col.get("postal code", col.get("postalcode", "")), "")) or None,
                    region=str(row.get(col.get("region", ""), "")) or None,
                    category=str(row.get(col.get("category", ""), "")) or None,
                    sub_category=str(row.get(col.get("sub-category", col.get("subcategory", "")), "")) or None,
                    product_id=str(row.get(col.get("product id", col.get("productid", "")), "")) or None,
                    cost_price=_safe_float(row.get(col.get("cost price", col.get("costprice", "")), None)),
                    list_price=_safe_float(row.get(col.get("list price", col.get("listprice", col.get("sales", ""))), None)),
                    quantity=_safe_int(row.get(col.get("quantity", ""), 1)) or 1,
                    discount_percent=_safe_float(row.get(col.get("discount", ""), 0)) or 0.0,
                    discount=_safe_float(row.get(col.get("discount amount", ""), 0)) or 0.0,
                    sale_price=_safe_float(row.get(col.get("sale price", col.get("sales", "")), None)),
                    profit=_safe_float(row.get(col.get("profit", ""), None)),
                    is_returned=_safe_bool(row.get(col.get("returned", col.get("is_returned", "")), False)),
                ))
            except Exception as e:
                logger.debug(f"Skipping order row: {e}")
                continue

        logger.info(f"Loaded {len(orders)} orders")
        return orders

    # ── Shipments ────────────────────────────────────────────────────────────

    def _load_shipments(self) -> List[Shipment]:
        """Load shipments from supply_chain.csv (or similar)."""
        candidates = [
            "supply_chain.csv",
            "shipments.csv",
            "dynamic_supply_chain_logistics_dataset.csv",
        ]
        df = self._find_and_read(candidates)
        if df is None:
            logger.warning("No shipments file found — generating sample data")
            return self._generate_sample_shipments()

        shipments: List[Shipment] = []
        col = {c.lower().strip(): c for c in df.columns}

        for _, row in df.iterrows():
            try:
                ship_id = str(row.get(col.get("shipment id", col.get("shipmentid", "")), uuid.uuid4()))
                ship_date = _safe_date(row.get(col.get("shipment date", col.get("shipmentdate", "")), None)) or datetime.now()
                exp_del = _safe_date(row.get(col.get("expected delivery", col.get("expecteddelivery", "")), None)) or datetime.now()

                shipments.append(Shipment(
                    shipment_id=ship_id,
                    product_id=str(row.get(col.get("product id", ""), "")) or None,
                    origin_port=str(row.get(col.get("origin port", col.get("origin", "")), "")) or None,
                    destination_port=str(row.get(col.get("destination port", col.get("destination", "")), "")) or None,
                    carrier=str(row.get(col.get("carrier", ""), "")) or None,
                    shipment_date=ship_date,
                    expected_delivery=exp_del,
                    actual_delivery=_safe_date(row.get(col.get("actual delivery", ""), None)),
                    quantity=_safe_int(row.get(col.get("quantity", ""), None)),
                    weight_kg=_safe_float(row.get(col.get("weight", col.get("weight kg", "")), None)),
                    freight_cost=_safe_float(row.get(col.get("freight cost", ""), None)),
                    insurance_cost=_safe_float(row.get(col.get("insurance cost", ""), None)),
                    customs_cost=_safe_float(row.get(col.get("customs cost", ""), None)),
                    status=str(row.get(col.get("status", ""), "")) or None,
                    delay_reason=str(row.get(col.get("delay reason", ""), "")) or None,
                ))
            except Exception as e:
                logger.debug(f"Skipping shipment row: {e}")
                continue

        if not shipments:
            return self._generate_sample_shipments()

        logger.info(f"Loaded {len(shipments)} shipments")
        return shipments

    # ── Inventory ────────────────────────────────────────────────────────────

    def _load_inventory(self) -> List[InventoryItem]:
        """Load inventory from inventory.csv (or similar)."""
        candidates = ["inventory.csv", "stock.csv"]
        df = self._find_and_read(candidates)
        if df is None:
            logger.warning("No inventory file found — generating sample data")
            return self._generate_sample_inventory()

        items: List[InventoryItem] = []
        col = {c.lower().strip(): c for c in df.columns}

        for _, row in df.iterrows():
            try:
                sku = str(row.get(col.get("sku", col.get("product id", "")), uuid.uuid4()))
                items.append(InventoryItem(
                    sku=sku,
                    product_id=str(row.get(col.get("product id", ""), "")) or None,
                    product_name=str(row.get(col.get("product name", ""), "")) or None,
                    category=str(row.get(col.get("category", ""), "")) or None,
                    warehouse_location=str(row.get(col.get("warehouse", col.get("location", "")), "")) or None,
                    stock_quantity=_safe_int(row.get(col.get("stock quantity", col.get("quantity", "")), 0)) or 0,
                    reorder_level=_safe_int(row.get(col.get("reorder level", col.get("reorder point", "")), None)),
                    reorder_quantity=_safe_int(row.get(col.get("reorder quantity", ""), None)),
                    unit_cost=_safe_float(row.get(col.get("unit cost", col.get("cost", "")), None)),
                    last_restock_date=_safe_date(row.get(col.get("last restock", ""), None)),
                    lead_time_days=_safe_int(row.get(col.get("lead time", col.get("lead time days", "")), None)),
                    supplier_id=str(row.get(col.get("supplier id", col.get("supplier", "")), "")) or None,
                ))
            except Exception as e:
                logger.debug(f"Skipping inventory row: {e}")
                continue

        if not items:
            return self._generate_sample_inventory()

        logger.info(f"Loaded {len(items)} inventory items")
        return items

    # ── Financial Transactions ───────────────────────────────────────────────

    def _load_transactions(self) -> List[FinancialTransaction]:
        """Load financial transactions from financial_data.csv (or similar)."""
        candidates = ["financial_data.csv", "transactions.csv", "finance.csv"]
        df = self._find_and_read(candidates)
        if df is None:
            logger.warning("No financial data file found — generating sample data")
            return self._generate_sample_transactions()

        txns: List[FinancialTransaction] = []
        col = {c.lower().strip(): c for c in df.columns}

        for _, row in df.iterrows():
            try:
                txn_id = str(row.get(col.get("transaction id", col.get("id", "")), uuid.uuid4()))
                txn_date = _safe_date(row.get(col.get("date", col.get("transaction date", "")), None)) or datetime.now()
                amount = _safe_float(row.get(col.get("amount", ""), 0)) or 0.0

                txns.append(FinancialTransaction(
                    transaction_id=txn_id,
                    transaction_date=txn_date,
                    transaction_type=str(row.get(col.get("type", col.get("transaction type", "")), "expense")),
                    category=str(row.get(col.get("category", ""), "")) or None,
                    subcategory=str(row.get(col.get("subcategory", ""), "")) or None,
                    amount=amount,
                    currency=str(row.get(col.get("currency", ""), "USD")) or "USD",
                    cost_center=str(row.get(col.get("cost center", ""), "")) or None,
                    business_unit=str(row.get(col.get("business unit", ""), "")) or None,
                    payment_method=str(row.get(col.get("payment method", ""), "")) or None,
                    vendor_id=str(row.get(col.get("vendor id", col.get("vendor", "")), "")) or None,
                    notes=str(row.get(col.get("notes", col.get("description", "")), "")) or None,
                ))
            except Exception as e:
                logger.debug(f"Skipping transaction row: {e}")
                continue

        if not txns:
            return self._generate_sample_transactions()

        logger.info(f"Loaded {len(txns)} transactions")
        return txns

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_and_read(self, candidates: List[str]) -> Optional[pd.DataFrame]:
        for name in candidates:
            path = self.data_dir / name
            if path.exists():
                return _read_csv_safe(str(path))
        return None

    # ── Sample data generators ───────────────────────────────────────────────

    def _generate_sample_shipments(self, n: int = 100) -> List[Shipment]:
        import random
        carriers = ["FedEx", "UPS", "DHL", "USPS", "Amazon Logistics"]
        statuses = ["delivered", "in_transit", "delayed", "pending"]
        shipments = []
        for i in range(n):
            ship_date = datetime(2024, random.randint(1, 12), random.randint(1, 28))
            exp_del = datetime(2024, min(ship_date.month + 1, 12), random.randint(1, 28))
            actual_del = exp_del if random.random() > 0.3 else None
            shipments.append(Shipment(
                shipment_id=f"SHP-{i+1:04d}",
                product_id=f"PROD-{random.randint(1, 50):03d}",
                origin_port=random.choice(["New York", "Los Angeles", "Chicago"]),
                destination_port=random.choice(["Miami", "Seattle", "Dallas"]),
                carrier=random.choice(carriers),
                shipment_date=ship_date,
                expected_delivery=exp_del,
                actual_delivery=actual_del,
                quantity=random.randint(1, 100),
                weight_kg=round(random.uniform(0.5, 50.0), 2),
                freight_cost=round(random.uniform(10, 500), 2),
                status=random.choice(statuses),
            ))
        return shipments

    def _generate_sample_inventory(self, n: int = 100) -> List[InventoryItem]:
        import random
        categories = ["Electronics", "Furniture", "Office Supplies", "Technology", "Clothing"]
        warehouses = ["WH-East", "WH-West", "WH-Central", "WH-North"]
        items = []
        for i in range(n):
            stock = random.randint(0, 500)
            reorder = random.randint(10, 50)
            items.append(InventoryItem(
                sku=f"SKU-{i+1:04d}",
                product_id=f"PROD-{i+1:03d}",
                product_name=f"Product {i+1}",
                category=random.choice(categories),
                warehouse_location=random.choice(warehouses),
                stock_quantity=stock,
                reorder_level=reorder,
                reorder_quantity=reorder * 2,
                unit_cost=round(random.uniform(5, 500), 2),
                lead_time_days=random.randint(1, 30),
                supplier_id=f"SUP-{random.randint(1, 20):03d}",
            ))
        return items

    def _generate_sample_transactions(self, n: int = 200) -> List[FinancialTransaction]:
        import random
        types = ["revenue", "expense", "cogs", "discount"]
        categories = ["Sales", "Marketing", "Operations", "HR", "IT", "Logistics"]
        txns = []
        for i in range(n):
            txn_type = random.choice(types)
            amount = round(random.uniform(100, 50000), 2)
            if txn_type == "expense":
                amount = -abs(amount)
            txns.append(FinancialTransaction(
                transaction_id=f"TXN-{i+1:05d}",
                transaction_date=datetime(2024, random.randint(1, 12), random.randint(1, 28)),
                transaction_type=txn_type,
                category=random.choice(categories),
                amount=amount,
                currency="USD",
                cost_center=f"CC-{random.randint(1, 10):02d}",
                business_unit=random.choice(["Supply Chain", "Finance", "Operations"]),
            ))
        return txns
