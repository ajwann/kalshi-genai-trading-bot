import pytest
from datetime import datetime, timedelta
from spending_limit import SpendingLimitTracker


def test_spending_limit_tracker_initialization():
    """Test that SpendingLimitTracker initializes with default values."""
    tracker = SpendingLimitTracker()
    assert tracker.limit_period_hours == 24
    assert tracker.max_spending_cents == 10000  # $100.00


def test_spending_limit_tracker_custom_values():
    """Test that SpendingLimitTracker accepts custom values."""
    tracker = SpendingLimitTracker(limit_period_hours=12, max_spending_cents=5000)
    assert tracker.limit_period_hours == 12
    assert tracker.max_spending_cents == 5000


def test_calculate_order_cost():
    """Test order cost calculation."""
    tracker = SpendingLimitTracker()
    assert tracker.calculate_order_cost(count=1, price_cents=75) == 75
    assert tracker.calculate_order_cost(count=2, price_cents=50) == 100
    assert tracker.calculate_order_cost(count=10, price_cents=25) == 250


def test_calculate_spending_from_orders_executed():
    """Test spending calculation from executed orders."""
    tracker = SpendingLimitTracker()
    
    orders = [
        {
            "order_id": "1",
            "status": "executed",
            "side": "yes",
            "taker_fill_cost": 75,
            "maker_fill_cost": 0,
            "remaining_count": 0
        },
        {
            "order_id": "2",
            "status": "executed",
            "side": "no",
            "taker_fill_cost": 0,
            "maker_fill_cost": 100,
            "remaining_count": 0
        },
        {
            "order_id": "3",
            "status": "executed",
            "side": "yes",
            "taker_fill_cost": 50,
            "maker_fill_cost": 30,
            "remaining_count": 0
        },
    ]
    
    total = tracker.calculate_spending_from_orders(orders)
    assert total == 75 + 100 + 80  # 255 cents


def test_calculate_spending_from_orders_resting():
    """Test spending calculation includes committed capital from resting orders."""
    tracker = SpendingLimitTracker()
    
    orders = [
        {
            "order_id": "1",
            "status": "resting",
            "side": "yes",
            "taker_fill_cost": 25,  # Partially filled
            "maker_fill_cost": 0,
            "remaining_count": 3,  # 3 contracts still pending
            "yes_price": 75
        },
    ]
    
    total = tracker.calculate_spending_from_orders(orders)
    # 25 (filled) + (3 * 75) (committed) = 25 + 225 = 250 cents
    assert total == 250


def test_calculate_spending_from_orders_canceled():
    """Test that canceled orders only count what was filled."""
    tracker = SpendingLimitTracker()
    
    orders = [
        {
            "order_id": "1",
            "status": "canceled",
            "side": "yes",
            "taker_fill_cost": 50,  # 2 contracts filled before cancel
            "maker_fill_cost": 0,
            "remaining_count": 3  # 3 contracts canceled, don't count
        },
    ]
    
    total = tracker.calculate_spending_from_orders(orders)
    assert total == 50  # Only filled portion


def test_can_place_order_within_limit():
    """Test that orders within limit are allowed."""
    tracker = SpendingLimitTracker(max_spending_cents=10000)  # $100 limit
    
    can_place, reason = tracker.can_place_order(
        order_cost_cents=5000,  # $50 order
        current_spending_cents=3000  # $30 already spent
    )
    
    assert can_place is True
    assert "Spending OK" in reason


def test_can_place_order_exceeds_limit():
    """Test that orders exceeding limit are blocked."""
    tracker = SpendingLimitTracker(max_spending_cents=10000)  # $100 limit
    
    can_place, reason = tracker.can_place_order(
        order_cost_cents=8000,  # $80 order
        current_spending_cents=3000  # $30 already spent = $110 total
    )
    
    assert can_place is False
    assert "exceed spending limit" in reason.lower()


def test_can_place_order_at_limit():
    """Test order at exact limit boundary."""
    tracker = SpendingLimitTracker(max_spending_cents=10000)  # $100 limit
    
    can_place, reason = tracker.can_place_order(
        order_cost_cents=7000,  # $70 order
        current_spending_cents=3000  # $30 already spent = $100 total
    )
    
    assert can_place is True  # At limit is OK


def test_calculate_current_spending_filters_by_time():
    """Test that calculate_current_spending filters orders by time."""
    tracker = SpendingLimitTracker(limit_period_hours=24)
    
    now = datetime.utcnow()
    old_time = (now - timedelta(hours=25)).isoformat() + "Z"  # 25 hours ago
    recent_time = (now - timedelta(hours=12)).isoformat() + "Z"  # 12 hours ago
    
    orders = [
        {
            "order_id": "1",
            "status": "executed",
            "created_time": old_time,  # Outside period
            "taker_fill_cost": 100,
            "maker_fill_cost": 0,
            "remaining_count": 0
        },
        {
            "order_id": "2",
            "status": "executed",
            "created_time": recent_time,  # Within period
            "taker_fill_cost": 200,
            "maker_fill_cost": 0,
            "remaining_count": 0
        },
    ]
    
    spending = tracker.calculate_current_spending(orders)
    assert spending == 200  # Only the recent order counts


def test_get_period_start_time():
    """Test period start time calculation."""
    tracker = SpendingLimitTracker(limit_period_hours=24)
    period_start = tracker.get_period_start_time()
    
    now = datetime.utcnow()
    expected_start = now - timedelta(hours=24)
    
    # Allow 1 second difference for execution time
    assert abs((period_start - expected_start).total_seconds()) < 1


def test_calculate_spending_mixed_order_statuses():
    """Test spending calculation with mixed order statuses."""
    tracker = SpendingLimitTracker()
    
    orders = [
        {
            "order_id": "1",
            "status": "executed",
            "taker_fill_cost": 100,
            "maker_fill_cost": 0,
            "remaining_count": 0
        },
        {
            "order_id": "2",
            "status": "resting",
            "side": "yes",
            "taker_fill_cost": 50,
            "maker_fill_cost": 0,
            "remaining_count": 2,
            "yes_price": 75
        },
        {
            "order_id": "3",
            "status": "canceled",
            "taker_fill_cost": 25,
            "maker_fill_cost": 0,
            "remaining_count": 5
        },
    ]
    
    total = tracker.calculate_spending_from_orders(orders)
    # 100 (executed) + 50 (filled) + 150 (committed) + 25 (canceled but filled) = 325 cents
    assert total == 325

