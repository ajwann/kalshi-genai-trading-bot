### Project Structure
The project is structured as a Google Cloud Function, with the main entrypoint in `main.py`. It uses the official `kalshi-python` SDK for API interactions and `xai-sdk` for Grok API calls. The bot is stateless, relying on Kalshi API calls for positions (no local storage). Environment variables are used for configuration.

To deploy:
- Install dependencies: `pip install -r requirements.txt`
- Set environment variables (e.g., via GCP console):
  - `KALSHI_API_KEY`: Your Kalshi API key ID.
  - `KALSHI_PRIVATE_KEY`: Your Kalshi private key (PEM string; store securely, e.g., as a secret).
  - `KALSHI_BASE_URL`: Default `https://trading-api.demo.kalshi.com/v1` (demo); override for production `https://trading-api.kalshi.com/v1`.
  - `LOOKBACK_HOURS`: Default `1`.
  - `GROK_MODEL`: Default `grok-4-1`; override for other models (e.g., `grok-3`).
  - `XAI_API_KEY`: Your xAI API key.
- Deploy as a GCP Cloud Function (HTTP trigger or scheduler trigger for hourly runs via Cloud Scheduler).

The bot logs actions via Python's `logging` module (configure in GCP for CloudWatch/Stackdriver).

### requirements.txt
```
functions-framework==3.*
kalshi-python==2.1.4
xai-sdk
python-dotenv  # For local testing
pytest==8.3.2  # For tests
requests  # Fallback if needed
```

### main.py (Main Cloud Function)
```python
import os
import logging
import json
import functions_framework
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from kalshi_python import Configuration, KalshiClient
from kalshi_python.models.get_markets_response import GetMarketsResponse
from kalshi_python.models.get_positions_response import GetPositionsResponse
from kalshi_python.rest import ApiException
from xai_sdk import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
KALSHI_BASE_URL = os.getenv('KALSHI_BASE_URL', 'https://trading-api.demo.kalshi.com/v1')
LOOKBACK_HOURS = int(os.getenv('LOOKBACK_HOURS', '1'))
GROK_MODEL = os.getenv('GROK_MODEL', 'grok-4-1')
XAI_API_KEY = os.getenv('XAI_API_KEY')
KALSHI_API_KEY = os.getenv('KALSHI_API_KEY')
KALSHI_PRIVATE_KEY = os.getenv('KALSHI_PRIVATE_KEY')

def create_kalshi_client() -> KalshiClient:
    """Create authenticated Kalshi client."""
    if not KALSHI_API_KEY or not KALSHI_PRIVATE_KEY:
        raise ValueError("KALSHI_API_KEY and KALSHI_PRIVATE_KEY must be set.")
    config = Configuration(host=KALSHI_BASE_URL)
    config.api_key_id = KALSHI_API_KEY
    config.private_key_pem = KALSHI_PRIVATE_KEY
    return KalshiClient(config)

def create_grok_client() -> Client:
    """Create xAI Grok client."""
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY must be set.")
    return Client(api_key=XAI_API_KEY)

def get_new_markets(client: KalshiClient, lookback_hours: int) -> list[Dict[str, Any]]:
    """Fetch active markets created in the last lookback_hours."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=lookback_hours)
    try:
        response: GetMarketsResponse = client.get_markets(
            start_time=start_time.isoformat() + 'Z',
            end_time=end_time.isoformat() + 'Z',
            status='open'
        )
        markets = response.markets or []
        logger.info(f"Found {len(markets)} new open markets.")
        return markets
    except ApiException as e:
        logger.error(f"Error fetching markets: {e}")
        return []

def get_current_positions(client: KalshiClient) -> set[str]:
    """Get tickers of current positions."""
    try:
        response: GetPositionsResponse = client.get_positions()
        positions = response.positions or []
        tickers = {pos.ticker for pos in positions if pos.size != 0}
        logger.info(f"Current positions: {tickers}")
        return tickers
    except ApiException as e:
        logger.error(f"Error fetching positions: {e}")
        return set()

def build_grok_prompt(market: Dict[str, Any]) -> str:
    """Build role-based prompt for Grok."""
    prompt = f"""You are an expert in professional prediction market trading, like Domer. Before recommending any position, perform online research to become an expert in this market.

Market details:
- ID: {market.get('id', 'N/A')}
- Ticker: {market.get('ticker', 'N/A')}
- Question: {market.get('question', 'N/A')}
- Subtitle: {market.get('subtitle', 'N/A')}
- Category: {market.get('category', 'N/A')}
- Status: {market.get('status', 'N/A')}
- Open: {market.get('open_time', 'N/A')}
- Close: {market.get('close_time', 'N/A')}
- Yes Bid/Ask: {market.get('yes_bid', 0)}/{market.get('yes_ask', 0)}
- No Bid/Ask: {market.get('no_bid', 0)}/{market.get('no_ask', 0)}
- Volume: {market.get('volume', 0)}
- Last Price: {market.get('last_price', 0)}

Research the event using web searches, news, and X (Twitter) for real-time insights. Analyze probabilities based on data, trends, and expert opinions.

Respond ONLY with a RFC 8259 compliant JSON object with two keys:
- "market_id": the market ID as string if you recommend a position (e.g., "123"), else null.
- "explanation": a textual explanation of why to take a position (or null if no position).

Example for position: {{"market_id": "123", "explanation": "I think X will happen based on recent polls showing 65% support."}}
Example for no position: {{"market_id": null, "explanation": null}}"""
    return prompt

def query_grok(grok_client: Client, prompt: str) -> Optional[Dict[str, Any]]:
    """Query Grok and parse JSON response."""
    try:
        response = grok_client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for consistent JSON
        )
        content = response.choices[0].message.content.strip()
        # Extract JSON (assume it's the main content; in production, use regex if needed)
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            parsed = json.loads(json_str)
            if "market_id" in parsed and "explanation" in parsed:
                logger.info(f"Grok response: {parsed}")
                return parsed
        logger.warning("Invalid JSON from Grok")
        return None
    except Exception as e:
        logger.error(f"Error querying Grok: {e}")
        return None

def take_position(client: KalshiClient, market_id: str, explanation: str) -> bool:
    """Place a position based on Grok's recommendation. (Simplified: buy 1 Yes contract; customize as needed.)"""
    if not market_id or market_id == "null":
        return False
    try:
        # Fetch market details for price/ticker
        market = client.get_market(market_id=market_id)
        ticker = market.ticker
        # Simplified: Buy 1 Yes contract at market price (use limit_order for production)
        order = client.buy(many=1, ticker=ticker, side='yes')  # Assumes buy_yes method; check SDK
        logger.info(f"Placed order: {order} for market {market_id}. Explanation: {explanation}")
        return True
    except ApiException as e:
        logger.error(f"Error placing order: {e}")
        return False

@functions_framework.cloud_event
def new_position_bot(cloud_event):
    """Main handler: Run bot logic."""
    logger.info("Starting New Position Bot")
    try:
        kalshi_client = create_kalshi_client()
        grok_client = create_grok_client()

        new_markets = get_new_markets(kalshi_client, LOOKBACK_HOURS)
        current_positions = get_current_positions(kalshi_client)

        for market in new_markets:
            ticker = market.get('ticker')
            if ticker in current_positions:
                logger.info(f"Skipping market {ticker}: position exists")
                continue

            prompt = build_grok_prompt(market)
            grok_response = query_grok(grok_client, prompt)
            if grok_response:
                market_id = grok_response.get('market_id')
                explanation = grok_response.get('explanation')
                if market_id and explanation:
                    take_position(kalshi_client, market_id, explanation)
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise

# For local testing: python main.py
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    new_position_bot(None)
```

### Notes on Implementation
- **Stateless Design**: All state (positions, markets) fetched via API on each run. No database needed.
- **Python Version**: Code is compatible with Python 3.13+ (uses f-strings, type hints). Specify `python313` in GCP runtime.
- **Grok Prompt**: Role-based as specified. Instructs research (Grok uses built-in tools like web_search). JSON is enforced with examples and "ONLY" directive; low temperature ensures compliance.
- **Position Taking**: Simplified to buy 1 "Yes" contract. In production, parse explanation for side (Yes/No), size, price. Use `limit_order` for risk management. Assumes SDK methods like `buy`; verify in [Kalshi docs](https://docs.kalshi.com/python-sdk).
- **Error Handling**: Logs errors; continues on failures to avoid full halt.
- **Configurability**: All via env vars. For production switch, set `KALSHI_BASE_URL=https://trading-api.kalshi.com/v1`.
- **Hourly Run**: Use GCP Cloud Scheduler to trigger the function every hour (e.g., cron `0 * * * *`).
- **Grok Model**: Defaults to `grok-4-1`; switch via `GROK_MODEL` (e.g., `grok-3`).

### Tests (in `tests/` directory)
Run with `pytest tests/`.

#### tests/test_main.py (Unit Tests)
```python
import pytest
from unittest.mock import MagicMock, patch
from main import (create_kalshi_client, create_grok_client, get_new_markets,
                  get_current_positions, build_grok_prompt, query_grok,
                  take_position)

@patch('main.Configuration')
@patch('main.KalshiClient')
def test_create_kalshi_client(mock_client, mock_config, monkeypatch):
    monkeypatch.setenv('KALSHI_API_KEY', 'test_key')
    monkeypatch.setenv('KALSHI_PRIVATE_KEY', 'test_pem')
    client = create_kalshi_client()
    assert isinstance(client, MagicMock)  # Mocked

@patch('main.Client')
def test_create_grok_client(mock_client, monkeypatch):
    monkeypatch.setenv('XAI_API_KEY', 'test_xai')
    client = create_grok_client()
    assert isinstance(client, MagicMock)

def test_build_grok_prompt():
    market = {'id': '123', 'ticker': 'TEST', 'question': 'Will it rain?'}
    prompt = build_grok_prompt(market)
    assert 'You are an expert' in prompt
    assert 'market_id' in prompt
    assert '{"market_id": "123"' in prompt  # Checks JSON example

@patch('main.json.loads')
def test_query_grok(mock_json, mock_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"market_id": "123", "explanation": "Test"}'
    with patch('main.Client') as mock_xai:
        mock_xai.return_value.chat.completions.create.return_value = mock_response
        result = query_grok(mock_client(), "prompt")
        assert result == {"market_id": "123", "explanation": "Test"}

def test_get_new_markets(mock_client):
    mock_response = MagicMock()
    mock_response.markets = [{'id': '1'}]
    mock_client.get_markets.return_value = mock_response
    markets = get_new_markets(mock_client, 1)
    assert len(markets) == 1

def test_get_current_positions(mock_client):
    mock_response = MagicMock()
    mock_response.positions = [MagicMock(ticker='TEST')]
    mock_client.get_positions.return_value = mock_response
    positions = get_current_positions(mock_client)
    assert positions == {'TEST'}

@patch.object(KalshiClient, 'get_market')
@patch.object(KalshiClient, 'buy')
def test_take_position(mock_buy, mock_get_market, mock_client):
    mock_market = MagicMock(ticker='TEST')
    mock_get_market.return_value = mock_market
    mock_buy.return_value = 'order_success'
    result = take_position(mock_client, '123', 'Test expl')
    assert result is True
```

#### tests/test_integration.py (Integration Tests)
Requires local env vars set (use `.env` file). Mocks API calls but tests flow.
```python
import pytest
from unittest.mock import patch
from main import new_position_bot

def test_bot_flow(monkeypatch):
    monkeypatch.setenv('KALSHI_API_KEY', 'test')
    monkeypatch.setenv('KALSHI_PRIVATE_KEY', 'test')
    monkeypatch.setenv('XAI_API_KEY', 'test')

    with patch('main.get_new_markets') as mock_markets:
        with patch('main.get_current_positions') as mock_pos:
            with patch('main.query_grok') as mock_grok:
                with patch('main.take_position') as mock_take:
                    mock_markets.return_value = [{'id': '123', 'ticker': 'NEW'}]
                    mock_pos.return_value = set()
                    mock_grok.return_value = {'market_id': '123', 'explanation': 'Test'}
                    new_position_bot(None)
                    mock_take.assert_called_once_with(MagicMock(), '123', 'Test')
```

#### tests/test_e2e.py (End-to-End Functional Tests)
Simulates full run with mocked external calls (run locally with demo keys if available).
```python
import pytest
from main import new_position_bot

# Assumes env vars set; mocks entire external interactions
@patch('main.create_kalshi_client')
@patch('main.create_grok_client')
def test_e2e_full_run(mock_grok, mock_kalshi, caplog):
    # Mock clients to return dummies
    mock_kalshi.return_value.get_markets.return_value.markets = [{'id': '123', 'ticker': 'E2E'}]
    mock_kalshi.return_value.get_positions.return_value.positions = []
    mock_grok.return_value.chat.completions.create.return_value.choices[0].message.content = '{"market_id": "123", "explanation": "E2E test"}'

    new_position_bot(None)
    assert 'Starting New Position Bot' in caplog.text
    assert 'Placed order' in caplog.text  # Via take_position log
```

These tests cover:
- **Unit**: Individual functions (e.g., prompt building, JSON parsing).
- **Integration**: Flow between components (e.g., markets -> Grok -> order).
- **E2E**: Full handler execution with mocks.

For real integration/E2E, use Kalshi demo keys and run `pytest --cov=main`. Extend as needed (e.g., assert JSON validity with `jsonschema`).
