from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SpendingLimitTracker:
    def __init__(self, limit_period_hours: int = 24, max_spending_cents: int = 10000):
        """
        Tracks spending over a time period.
        
        Args:
            limit_period_hours: Time window in hours (default 24)
            max_spending_cents: Maximum spending in cents (default $100.00)
        """
        self.limit_period_hours = limit_period_hours
        self.max_spending_cents = max_spending_cents
    
    def calculate_spending_from_orders(self, orders: List[Dict]) -> int:
        """
        Calculate total spending from a list of orders.
        
        For executed orders: Uses actual fill cost (taker_fill_cost + maker_fill_cost)
        For resting orders: Uses remaining_count * price (committed capital)
        For canceled orders: Counts only what was filled before cancellation
        
        Args:
            orders: List of order dictionaries from Kalshi API
            
        Returns:
            Total spending in cents (actual + committed)
        """
        total_spending = 0
        
        for order in orders:
            status = order.get("status", "unknown")
            
            # Count actual money spent (from fills) - sum both taker and maker
            taker_fill_cost = order.get("taker_fill_cost", 0) or 0
            maker_fill_cost = order.get("maker_fill_cost", 0) or 0
            fill_cost = taker_fill_cost + maker_fill_cost
            total_spending += fill_cost
            
            # For resting orders, also count committed capital
            if status == "resting":
                remaining_count = order.get("remaining_count", 0)
                side = order.get("side", "yes")
                
                # Get price based on side
                if side == "yes":
                    price = order.get("yes_price", 0)
                else:
                    price = order.get("no_price", 0)
                
                committed_cost = remaining_count * price
                total_spending += committed_cost
                
                logger.debug(
                    f"Resting order {order.get('order_id', 'unknown')}: "
                    f"spent={fill_cost}, committed={committed_cost} cents"
                )
            else:
                logger.debug(
                    f"{status} order {order.get('order_id', 'unknown')}: spent={fill_cost} cents"
                )
        
        return total_spending
    
    def get_period_start_time(self) -> datetime:
        """Get the start time for the spending limit period."""
        return datetime.utcnow() - timedelta(hours=self.limit_period_hours)
    
    def calculate_current_spending(self, orders: List[Dict]) -> int:
        """
        Calculate spending from orders within the limit period.
        
        Args:
            orders: All orders (will be filtered by time)
            
        Returns:
            Total spending in cents within the period
        """
        period_start = self.get_period_start_time()
        
        # Filter orders within the period
        recent_orders = []
        for order in orders:
            order_time_str = order.get("created_time")
            if order_time_str:
                try:
                    if isinstance(order_time_str, (int, float)):
                        order_time = datetime.fromtimestamp(order_time_str / 1000)
                    else:
                        order_time = datetime.fromisoformat(order_time_str.replace("Z", "+00:00"))
                        order_time = order_time.replace(tzinfo=None)
                    
                    if order_time >= period_start:
                        recent_orders.append(order)
                except Exception as e:
                    logger.warning(f"Failed to parse order time: {e}")
                    continue
        
        return self.calculate_spending_from_orders(recent_orders)
    
    def can_place_order(self, order_cost_cents: int, current_spending_cents: int) -> tuple[bool, str]:
        """
        Check if a new order can be placed without exceeding the limit.
        
        Args:
            order_cost_cents: Cost of the new order in cents
            current_spending_cents: Current spending in the period in cents
            
        Returns:
            Tuple of (can_place: bool, reason: str)
        """
        total_after_order = current_spending_cents + order_cost_cents
        
        if total_after_order > self.max_spending_cents:
            remaining = self.max_spending_cents - current_spending_cents
            return False, (
                f"Order would exceed spending limit. "
                f"Current: ${current_spending_cents/100:.2f}, "
                f"Limit: ${self.max_spending_cents/100:.2f}, "
                f"Remaining: ${remaining/100:.2f}"
            )
        
        return True, (
            f"Spending OK. Current: ${current_spending_cents/100:.2f}, "
            f"After order: ${total_after_order/100:.2f}, "
            f"Limit: ${self.max_spending_cents/100:.2f}"
        )
    
    def calculate_order_cost(self, count: int, price_cents: int) -> int:
        """Calculate the cost of an order in cents."""
        return count * price_cents

