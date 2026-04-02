from unittest.mock import MagicMock, patch

from grok_client import GrokClient


def test_analyze_market_success():
    client = GrokClient("api_key")
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "{\"ticker\": \"ABC\", \"explanation\": \"Good logic\"}"
                }
            }
        ]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        result = client.analyze_market({"ticker": "ABC"})
        assert result["ticker"] == "ABC"
        assert result["explanation"] == "Good logic"


def test_analyze_market_null():
    client = GrokClient("api_key")
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "{\"ticker\": null, \"explanation\": null}"
                }
            }
        ]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        result = client.analyze_market({"ticker": "ABC"})
        assert result["ticker"] is None


def test_analyze_market_includes_settlement_rules_in_prompt():
    """When settlement_rules are provided the user prompt must contain them."""
    client = GrokClient("api_key")
    rules_text = (
        "--- Settlement Rules (authoritative for resolution) ---\n"
        "Yes outcome: Above 3.5%\n"
        "Primary rules: Resolves Yes if CPI-U > 3.5%.\n"
        "--- End Settlement Rules ---"
    )

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": '{"ticker": "CPI-25APR-T3.5", "side": "yes", '
                    '"explanation": "CPI expected above 3.5%"}'
                }
            }
        ]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = MagicMock()

        client.analyze_market(
            {"ticker": "CPI-25APR-T3.5", "title": "CPI Above 3.5%"},
            settlement_rules=rules_text,
        )

        sent_payload = mock_post.call_args[1]["json"]
        user_message = sent_payload["messages"][1]["content"]
        assert "Settlement Rules" in user_message
        assert "Resolves Yes if CPI-U > 3.5%" in user_message
        assert "Above 3.5%" in user_message


def test_analyze_market_no_rules_still_works():
    """Passing empty settlement_rules should not add the rules block."""
    client = GrokClient("api_key")
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": '{"ticker": null, "side": null, "explanation": "skip"}'
                }
            }
        ]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = MagicMock()

        client.analyze_market({"ticker": "ABC"}, settlement_rules="")

        sent_payload = mock_post.call_args[1]["json"]
        user_message = sent_payload["messages"][1]["content"]
        assert "Settlement Rules" not in user_message
