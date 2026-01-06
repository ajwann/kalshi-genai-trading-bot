import pytest
from unittest.mock import patch
from new_position_bot.grok_client import GrokClient

def test_analyze_market_success():
    client = GrokClient("api_key")
    mock_response = {
        "choices": [{
            "message": {
                "content": "{\"ticker\": \"ABC\", \"explanation\": \"Good logic\"}"
            }
        }]
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
        "choices": [{
            "message": {
                "content": "{\"ticker\": null, \"explanation\": null}"
            }
        }]
    }
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        result = client.analyze_market({"ticker": "ABC"})
        assert result["ticker"] is None
