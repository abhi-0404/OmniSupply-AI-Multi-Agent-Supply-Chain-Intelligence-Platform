"""
Pydantic data models for OmniSupply platform.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class Order(BaseModel):
    """Order data model"""
    order_id: str
    order_date: datetime
    ship_mode: Optional[str] = None
    segment: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    region: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    product_id: Optional[str] = None
    cost_price: Optional[float] = None
    list_price: Optional[float] = None
    quantity: Optional[int] = 1
    discount_percent: Optional[float] = 0.0
    discount: Optional[float] = 0.0
    sale_price: Optional[float] = None
    profit: Optional[float] = None
    is_returned: Optional[bool] = False

    class Config:
        arbitrary_types_allowed = True


class Shipment(BaseModel):
    """Shipment data model"""
    shipment_id: str
    product_id: Optional[str] = None
    origin_port: Optional[str] = None
    destination_port: Optional[str] = None
    carrier: Optional[str] = None
    shipment_date: datetime
    expected_delivery: datetime
    actual_delivery: Optional[datetime] = None
    quantity: Optional[int] = None
    weight_kg: Optional[float] = None
    freight_cost: Optional[float] = None
    insurance_cost: Optional[float] = None
    customs_cost: Optional[float] = None
    status: Optional[str] = None
    delay_reason: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class InventoryItem(BaseModel):
    """Inventory item data model"""
    sku: str
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
    warehouse_location: Optional[str] = None
    stock_quantity: int = 0
    reorder_level: Optional[int] = None
    reorder_quantity: Optional[int] = None
    unit_cost: Optional[float] = None
    last_restock_date: Optional[datetime] = None
    lead_time_days: Optional[int] = None
    supplier_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class FinancialTransaction(BaseModel):
    """Financial transaction data model"""
    transaction_id: str
    transaction_date: datetime
    transaction_type: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    amount: float
    currency: str = "USD"
    cost_center: Optional[str] = None
    business_unit: Optional[str] = None
    payment_method: Optional[str] = None
    vendor_id: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class AgentResult(BaseModel):
    """Result from an agent execution"""
    agent_name: str
    query: str
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    raw_data: Optional[Any] = None
    execution_time_ms: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class RiskAssessment(BaseModel):
    """Risk assessment result"""
    overall_score: float = 0.0
    overall_severity: str = "LOW"
    delivery_risk: float = 0.0
    inventory_risk: float = 0.0
    quality_risk: float = 0.0
    financial_risk: float = 0.0
    top_risks: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class DataQualityResult(BaseModel):
    """Data quality check result"""
    dataset_name: str
    status: str = "PASSED"
    issues_found: int = 0
    issues: List[str] = Field(default_factory=list)
    records_checked: int = 0
    records_valid: int = 0
