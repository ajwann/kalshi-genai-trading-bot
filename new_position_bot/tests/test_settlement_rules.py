from unittest.mock import MagicMock, patch

import pytest

from main import _derive_series_ticker, build_settlement_rules


class TestDeriveSeriesTicker:
    def test_simple_series_date(self):
        assert _derive_series_ticker("CPI-25APR") == "CPI"

    def test_series_with_multi_segment_date(self):
        assert _derive_series_ticker("KXBTC-26APR02") == "KXBTC"

    def test_no_date_suffix(self):
        assert _derive_series_ticker("ONLYSERIES") == "ONLYSERIES"

    def test_alphanumeric_series_prefix(self):
        assert _derive_series_ticker("INX500-26APR02") == "INX500"

    def test_single_part(self):
        assert _derive_series_ticker("SOLO") == "SOLO"


class TestBuildSettlementRules:
    @pytest.fixture
    def mock_kalshi(self):
        client = MagicMock()
        return client

    def test_builds_full_rules_block(self, mock_kalshi):
        mock_kalshi.get_market.return_value = {
            "ticker": "CPI-25APR-T3.5",
            "event_ticker": "CPI-25APR",
            "yes_sub_title": "Above 3.5%",
            "no_sub_title": "3.5% or below",
            "rules_primary": "Resolves Yes if CPI-U exceeds 3.5%.",
            "rules_secondary": "Based on seasonally adjusted data.",
        }
        mock_kalshi.get_series.return_value = {
            "ticker": "CPI",
            "frequency": "monthly",
            "settlement_sources": [
                {"name": "Bureau of Labor Statistics", "url": "https://bls.gov/cpi/"}
            ],
            "contract_url": "https://kalshi.com/contracts/cpi",
        }

        cache = {}
        result = build_settlement_rules(mock_kalshi, "CPI-25APR-T3.5", cache)

        assert "Above 3.5%" in result
        assert "3.5% or below" in result
        assert "Resolves Yes if CPI-U exceeds 3.5%" in result
        assert "seasonally adjusted" in result
        assert "Bureau of Labor Statistics" in result
        assert "monthly" in result
        assert "https://kalshi.com/contracts/cpi" in result
        assert "CPI" in cache

    def test_caches_series_across_calls(self, mock_kalshi):
        mock_kalshi.get_market.return_value = {
            "ticker": "CPI-25APR-T3.5",
            "event_ticker": "CPI-25APR",
            "yes_sub_title": "Above 3.5%",
            "no_sub_title": "3.5% or below",
            "rules_primary": "Resolves Yes.",
            "rules_secondary": "",
        }

        cache = {
            "CPI": {
                "settlement_sources": [{"name": "BLS", "url": ""}],
                "frequency": "monthly",
            }
        }

        build_settlement_rules(mock_kalshi, "CPI-25APR-T3.5", cache)

        mock_kalshi.get_series.assert_not_called()

    def test_returns_empty_when_get_market_fails(self, mock_kalshi):
        mock_kalshi.get_market.side_effect = Exception("API error")

        result = build_settlement_rules(mock_kalshi, "BAD-TICKER", {})

        assert result == ""

    def test_graceful_when_series_fails(self, mock_kalshi):
        mock_kalshi.get_market.return_value = {
            "ticker": "NEW-26APR",
            "event_ticker": "NEW-26APR",
            "yes_sub_title": "Yes",
            "no_sub_title": "No",
            "rules_primary": "Some rule.",
            "rules_secondary": "",
        }
        mock_kalshi.get_series.side_effect = Exception("Series not found")

        cache = {}
        result = build_settlement_rules(mock_kalshi, "NEW-26APR", cache)

        assert "Some rule." in result
        assert "Settlement sources" not in result
        assert "NEW" in cache
