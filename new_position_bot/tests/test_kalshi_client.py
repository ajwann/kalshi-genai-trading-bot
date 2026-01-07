import pytest
from unittest.mock import MagicMock, patch, ANY
from kalshi_client import KalshiClient


@pytest.fixture
def mock_kalshi():
    dummy_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpQIBAAKCAQEA...\n"
        "-----END RSA PRIVATE KEY-----"
    )

    with patch("kalshi_client.load_pem_private_key") as mock_load:
        mock_load.return_value = MagicMock()
        client = KalshiClient("https://test.api", "key_id", dummy_key)
        client.private_key.sign.return_value = b"signature"
        return client


def test_get_active_markets(mock_kalshi):
    with patch.object(mock_kalshi, "_request") as mock_req:
        mock_req.return_value = {
            "markets": [
                {"ticker": "OLD", "created_time": "2020-01-01T00:00:00Z"},
                {"ticker": "NEW", "created_time": "2099-01-01T00:00:00Z"},
            ]
        }

        markets = mock_kalshi.get_active_markets(created_after_hours=1)

        assert len(markets) == 1
        assert markets[0]["ticker"] == "NEW"


def test_create_market_order_yes_price(mock_kalshi):
    with patch.object(mock_kalshi, "_request") as mock_req:
        mock_req.return_value = {"order_id": "123"}

        resp = mock_kalshi.create_market_order(
            ticker="TICKER",
            side="yes",
            count=1,
            yes_price=12,
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
                "client_order_id": ANY,
            },
        )


def test_create_market_order_accepts_yes_and_no_price(mock_kalshi):
    with patch.object(mock_kalshi, "_request") as mock_req:
        mock_req.return_value = {"order_id": "456"}

        resp = mock_kalshi.create_market_order(
            ticker="TICKER",
            side="no",
            count=2,
            yes_price=40,
            no_price=60,
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
                "client_order_id": ANY,
            },
        )


def test_create_market_order_includes_bot_identifier(mock_kalshi):
    with patch.object(mock_kalshi, "_request") as mock_req:
        mock_req.return_value = {"order_id": "789"}

        resp = mock_kalshi.create_market_order(
            ticker="TICKER",
            side="yes",
            count=1,
            price=75,
            bot_identifier="TEST_BOT"
        )

        assert resp == {"order_id": "789"}
        
        # Check that client_order_id starts with bot_identifier
        call_args = mock_req.call_args
        client_order_id = call_args[1]["data"]["client_order_id"]
        assert client_order_id.startswith("TEST_BOT-")


def test_get_orders_filters_by_bot_identifier(mock_kalshi):
    with patch.object(mock_kalshi, "_request") as mock_req:
        mock_req.return_value = {
            "orders": [
                {
                    "order_id": "1",
                    "client_order_id": "NEW_POSITION_BOT-123456",
                    "ticker": "TICKER1",
                    "status": "executed"
                },
                {
                    "order_id": "2",
                    "client_order_id": "manual-789012",
                    "ticker": "TICKER2",
                    "status": "executed"
                },
                {
                    "order_id": "3",
                    "client_order_id": "NEW_POSITION_BOT-345678",
                    "ticker": "TICKER3",
                    "status": "resting"
                },
            ],
            "cursor": None
        }

        orders = mock_kalshi.get_orders(bot_identifier="NEW_POSITION_BOT")

        assert len(orders) == 2
        assert all(order["client_order_id"].startswith("NEW_POSITION_BOT") for order in orders)
        assert orders[0]["order_id"] == "1"
        assert orders[1]["order_id"] == "3"


def test_get_orders_returns_all_orders_when_no_identifier(mock_kalshi):
    with patch.object(mock_kalshi, "_request") as mock_req:
        mock_req.return_value = {
            "orders": [
                {
                    "order_id": "1",
                    "client_order_id": "BOT-123",
                    "status": "executed"
                },
                {
                    "order_id": "2",
                    "client_order_id": "manual-456",
                    "status": "executed"
                },
            ],
            "cursor": None
        }

        orders = mock_kalshi.get_orders(bot_identifier=None)

        assert len(orders) == 2
