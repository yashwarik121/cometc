# signing off, hire me, yash warik!
"""
Order lookup tool with strict field stripping at the boundary.
Internal fields (email, address, notes, risk_score) are NEVER serialized
into the result the LLM sees.
"""

import json
import re
from pathlib import Path
from typing import Union

from src.config import ORDERS_FILE
from src.models import SafeOrderResult, OrderLookupError, OrderItem, OrderStatus


class OrderStore:
    """Loads and serves order data with sanitized output."""

    def __init__(self, orders_path: Path | None = None):
        self._orders: dict[str, dict] = {}
        self._load(orders_path or ORDERS_FILE)

    def _load(self, path: Path):
        """Load orders from JSON file, indexed by normalized ID."""
        with open(path, "r") as f:
            raw_orders = json.load(f)
        for order in raw_orders:
            normalized = self._normalize_id(order["order_id"])
            self._orders[normalized] = order

    @staticmethod
    def _normalize_id(order_id: str) -> str:
        """
        Normalize an order ID to canonical form 'AR-XXXXX'.
        Handles: AR-12345, ar-12345, AR 12345, #AR-12345, 12345,
        leading/trailing whitespace, mixed case.
        """
        cleaned = order_id.strip().upper()
        # Remove leading # or other prefixes
        cleaned = re.sub(r'^[#\s]+', '', cleaned)
        # Try to extract the AR-XXXXX pattern
        match = re.search(r'(AR[\s\-]?\d{5})', cleaned)
        if match:
            raw = match.group(1)
            # Normalize to AR-XXXXX
            digits = re.search(r'\d{5}', raw)
            if digits:
                return f"AR-{digits.group(0)}"
        # Try bare 5-digit number
        match = re.search(r'(\d{5})', cleaned)
        if match:
            return f"AR-{match.group(1)}"
        # Return cleaned as-is for error reporting
        return cleaned

    def lookup(self, raw_order_id: str) -> Union[SafeOrderResult, OrderLookupError]:
        """
        Look up an order by ID. Returns ONLY safe fields.
        Internal fields are stripped at this boundary — they never reach
        the serialized output.
        """
        if not raw_order_id or not raw_order_id.strip():
            return OrderLookupError(
                error="No order ID provided. Please provide an order ID (e.g., AR-12345).",
                order_id=""
            )

        normalized = self._normalize_id(raw_order_id)
        order = self._orders.get(normalized)

        if order is None:
            return OrderLookupError(
                error=f"Order {normalized} was not found. Please double-check the order ID and try again.",
                order_id=normalized
            )

        # Build safe result — only allowed fields, status-consistent
        status = OrderStatus(order["status"])
        items = [OrderItem(**item) for item in order["items"]]

        # Status-consistent field logic
        tracking = None
        carrier = None
        estimated_delivery = None
        actual_delivery = None
        cancellation_reason = None
        return_reason = None

        if status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED, OrderStatus.RETURN_REQUESTED):
            tracking = order.get("tracking_number")
            carrier = order.get("carrier")

        if status == OrderStatus.SHIPPED:
            estimated_delivery = order.get("estimated_delivery")

        if status == OrderStatus.DELIVERED:
            actual_delivery = order.get("actual_delivery")

        if status == OrderStatus.CANCELLED:
            cancellation_reason = order.get("cancellation_reason")

        if status in (OrderStatus.RETURNED, OrderStatus.RETURN_REQUESTED):
            return_reason = order.get("return_reason")

        return SafeOrderResult(
            order_id=normalized,
            customer_name=order["customer_name"],
            items=items,
            order_total=order["order_total"],
            order_date=order["order_date"],
            status=status,
            tracking_number=tracking,
            carrier=carrier,
            estimated_delivery=estimated_delivery,
            actual_delivery=actual_delivery,
            payment_method=order.get("payment_method"),
            cancellation_reason=cancellation_reason,
            return_reason=return_reason,
        )


# Module-level singleton (lazy init)
_store: OrderStore | None = None


def get_order_store() -> OrderStore:
    global _store
    if _store is None:
        _store = OrderStore()
    return _store


def order_lookup(order_id: str) -> dict:
    """
    Callable tool function for the agent.
    Returns a dict (serializable) with only safe fields.
    """
    store = get_order_store()
    result = store.lookup(order_id)
    return result.model_dump()
