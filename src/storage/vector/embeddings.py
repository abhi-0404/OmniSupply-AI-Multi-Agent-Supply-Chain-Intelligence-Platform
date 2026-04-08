"""
Embedding generation for semantic search in OmniSupply platform.
Uses Google Gemini embeddings via langchain-google-genai.
"""

import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings using Google Gemini text-embedding-004"""

    def __init__(
        self,
        model: str = "models/text-embedding-004",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.dimension = 768  # text-embedding-004 = 768 dims

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model,
                google_api_key=self.api_key
            )
            logger.info(f"Initialized Gemini embedding service: {model} ({self.dimension}d)")
        except Exception as e:
            logger.warning(f"Could not init Gemini embeddings: {e} — using dummy embeddings")
            self._embeddings = None

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return [0.0] * self.dimension
        if self._embeddings is None:
            return self._dummy_embedding(text)
        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return self._dummy_embedding(text)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        if self._embeddings is None:
            return [self._dummy_embedding(t) for t in texts]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embs = self._embeddings.embed_documents(batch)
                embeddings.extend(batch_embs)
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                embeddings.extend([self._dummy_embedding(t) for t in batch])
            logger.info(f"Embedded batch {i // batch_size + 1} ({len(batch)} texts)")
        return embeddings

    def _dummy_embedding(self, text: str) -> List[float]:
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(((h >> i) & 0xFF) / 255.0) for i in range(self.dimension)]


class DocumentPreprocessor:
    """Prepare documents for embedding"""

    @staticmethod
    def create_order_document(order: dict) -> str:
        return (
            f"Order {order.get('order_id')} in {order.get('category')} category. "
            f"Segment: {order.get('segment')}. Region: {order.get('region')}. "
            f"Product: {order.get('product_id')}. Sale: ${order.get('sale_price')}, "
            f"Profit: ${order.get('profit')}."
        )

    @staticmethod
    def create_shipment_document(shipment: dict) -> str:
        return (
            f"Shipment {shipment.get('shipment_id')} via {shipment.get('carrier')}. "
            f"Status: {shipment.get('status')}. "
            f"Freight: ${shipment.get('freight_cost')}."
        )

    @staticmethod
    def create_inventory_document(item: dict) -> str:
        return (
            f"SKU {item.get('sku')}: {item.get('product_name')}. "
            f"Stock: {item.get('stock_quantity')} units. "
            f"Reorder level: {item.get('reorder_level')}."
        )

    @staticmethod
    def create_transaction_document(txn: dict) -> str:
        return (
            f"Transaction {txn.get('transaction_id')} - {txn.get('transaction_type')}. "
            f"Amount: ${txn.get('amount')} {txn.get('currency')}. "
            f"Category: {txn.get('category')}."
        )

    @staticmethod
    def create_report_document(report: dict) -> str:
        return (
            f"Report: {report.get('report_type')}. "
            f"Summary: {str(report.get('summary', ''))[:300]}"
        )

    @staticmethod
    def create_alert_document(alert: dict) -> str:
        return (
            f"Alert: {alert.get('title')} ({alert.get('severity')}). "
            f"Description: {alert.get('description')}."
        )
