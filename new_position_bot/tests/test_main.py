from unittest.mock import ANY, MagicMock, patch

from main import get_affordable_order_count, get_kalshi_base_url, parse_args


def test_live_flag_selects_production_mode():
    assert parse_args([]).live is False
    assert parse_args(["--live"]).live is True


def test_live_mode_selects_production_api(monkeypatch):
    monkeypatch.delenv("KALSHI_BASE_URL", raising=False)

    assert get_kalshi_base_url(live=False) == "https://demo-api.kalshi.co"
    assert get_kalshi_base_url(live=True) == "https://api.elections.kalshi.com"


def test_zero_price_has_no_affordable_contracts():
    assert (
        get_affordable_order_count(
            requested_count=10,
            order_price=0,
            available_funds=1000,
            spending_limit_remaining=1000,
        )
        == 0
    )


def test_bot_uses_grok_contract_count_and_available_funds(monkeypatch):
    monkeypatch.setenv("KALSHI_API_KEY", "kalshi-key")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    monkeypatch.setenv("SPENDING_LIMIT_CENTS", "4000")

    market = {
        "ticker": "ABC",
        "title": "Example",
        "yes_ask": 25,
        "no_ask": 75,
    }
    kalshi = MagicMock()
    kalshi.get_orders.return_value = []
    kalshi.get_balance.return_value = 12500
    kalshi.get_active_markets.return_value = [market]
    kalshi.get_positions.return_value = []
    kalshi.get_market.return_value = {}

    grok = MagicMock()
    grok.analyze_market.return_value = {
        "ticker": "ABC",
        "side": "yes",
        "count": 8,
        "explanation": "High confidence",
    }

    with (
        patch("main.get_private_key", return_value="private-key"),
        patch("main.KalshiClient", return_value=kalshi),
        patch("main.GrokClient", return_value=grok),
    ):
        from main import run_bot_logic

        assert run_bot_logic() == ("Run Complete", 200)

    grok.analyze_market.assert_called_once_with(
        market,
        settlement_rules=ANY,
        available_funds_cents=12500,
        spending_limit_remaining_cents=4000,
    )
    kalshi.create_market_order.assert_called_once_with(
        "ABC",
        side="yes",
        count=8,
        price=25,
        bot_identifier="NEW_POSITION_BOT",
    )


def test_bot_caps_contract_count_to_affordable_funds(monkeypatch):
    monkeypatch.setenv("KALSHI_API_KEY", "kalshi-key")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    monkeypatch.setenv("SPENDING_LIMIT_CENTS", "4000")

    market = {
        "ticker": "ABC",
        "title": "Example",
        "yes_ask": 25,
        "no_ask": 75,
    }
    kalshi = MagicMock()
    kalshi.get_orders.return_value = []
    kalshi.get_balance.return_value = 1000
    kalshi.get_active_markets.return_value = [market]
    kalshi.get_positions.return_value = []
    kalshi.get_market.return_value = {}

    grok = MagicMock()
    grok.analyze_market.return_value = {
        "ticker": "ABC",
        "side": "yes",
        "count": 500,
        "explanation": "High confidence",
    }

    with (
        patch("main.get_private_key", return_value="private-key"),
        patch("main.KalshiClient", return_value=kalshi),
        patch("main.GrokClient", return_value=grok),
    ):
        from main import run_bot_logic

        assert run_bot_logic() == ("Run Complete", 200)

    kalshi.create_market_order.assert_called_once_with(
        "ABC",
        side="yes",
        count=40,
        price=25,
        bot_identifier="NEW_POSITION_BOT",
    )
