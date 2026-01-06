import pytest
from unittest.mock import MagicMock, patch, ANY
from new_position_bot.kalshi_client import KalshiClient

@pytest.fixture
def mock_kalshi():
    # Use a dummy key for testing initialization
    dummy_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpQIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    # We mock load_pem_private_key to avoid needing a real valid key in tests
    with patch("new_position_bot.kalshi_client.load_pem_private_key") as mock_load:
        mock_load.return_value = MagicMock()
        client = KalshiClient("https://test.api", "key_id", dummy_key)
        # Mock the sign method specifically
        client.private_key.sign.return_value = b"signature"
        return client

def test_get_active_markets(mock_kalshi):
    with patch.object(mock_kalshi, '_request') as mock_req:
        mock_req.return_value = {
            "markets": [
                {"ticker": "OLD", "created_time": "2020-01-01T00:00:00Z"},
                {"ticker": "NEW", "created_time": "2099-01-01T00:00:00Z"} 
            ]
        }
        # In this logic, NEW should be picked up as we check "created in past hour"
        # Since we hardcoded 2099, it's definitely "after" 1 hour ago.
        
        markets = mock_kalshi.get_active_markets(created_after_hours=1)
        assert len(markets) == 1
        assert markets[0]['ticker'] == "NEW"

def test_create_market_order(mock_kalshi):
    with patch.object(mock_kalshi, '_request') as mock_req:
        mock_req.return_value = {"order_id": "123"}

        resp = mock_kalshi.create_market_order(
            ticker="TICKER",
            side="yes",
            count=1,
            price=12
        )

        assert resp == {"order_id": "123"}

        mock_req.assert_called_with(
            "POST",
            "/portfolio/orders",
            data={
                "ticker": "TICKER",
                "action": "buy",
                "type": "market",
                "side": "yes",
                "count": 1,
                "yes_price": 12,
                "cancel_order_on_pause": True,
                "client_order_id": ANY
            }
        )

def test_create_market_order_accepts_yes_and_no_price(mock_kalshi):
    with patch.object(mock_kalshi, '_request') as mock_req:
        mock_req.return_value = {"order_id": "456"}

        resp = mock_kalshi.create_market_order(
            "TICKER",
            side="no",
            count=2,
            yes_price=40,
            no_price=60
        )

        assert resp == {"order_id": "456"}

        mock_req.assert_called_with(
            "POST",
            "/portfolio/orders",
            data={
                "ticker": "TICKER",
                "action": "buy",
                "type": "market",
                "side": "no",
                "count": 2,
                "yes_price": 40,
                "no_price": 60,
                "cancel_order_on_pause": True,
                "client_order_id": ANY
            }
        )

