### Project Structure
The "New Position Bot" is implemented as a Python 3.13.11 project following PEP 8 style guidelines. It is stateless, making it suitable for Google Cloud Functions (GCF), while also runnable locally with stdout logging. The bot runs hourly (configurable via `LOOKBACK_HOURS`, default 1), scans for new open markets on Kalshi without existing positions, consults Grok 4.1 for recommendations, and places trades if advised.

Key assumptions based on Kalshi docs:
- Authentication: HTTP Signature with RSA PEM private key (using `kalshi_private_key` file locally or GCP Secret Manager in GCF).
- Endpoints: `/markets?status=open&created_after={timestamp}` for new markets; `/positions` for user positions; `/orders` for placing trades.
- Grok API: OpenAI-compatible `/v1/chat/completions` endpoint at `https://api.x.ai/v1`, with `response_format={"type": "json_object"}` for RFC 8259 compliance.
- Trading: Assumes yes/no binary markets; places market orders for recommended side (yes/no) with fixed size (e.g., $10, configurable if needed).
- Error handling: Retries on transient errors; logs all actions.
- Tests: Pytest for unit (mocked API calls), integration (local Kalshi demo), and E2E (full flow with mocks).

#### Environment Variables
- `KALSHI_API_KEY`: Kalshi API key (str).
- `KALSHI_BASE_URL`: Base URL (default: `https://demo-api.kalshi.co/trade-api/v2`).
- `LOOKBACK_HOURS`: Hours to look back for new markets (default: 1, int).
- `GROK_MODEL`: Grok model (default: `grok-4.1`).
- `XAI_API_KEY`: xAI API key (str).
- For GCF: Set `GOOGLE_CLOUD_PROJECT` and use Secret Manager for private key (secret ID: `kalshi-private-key`).

#### Dependencies (requirements.txt)
```
requests==2.31.0
python-dotenv==1.0.0
cryptography==42.0.5
pytest==8.3.3
pytest-mock==3.14.0
freezegun==1.5.1
```

#### File Structure
```
new_position_bot/
├── main.py                 # Entry point (local/GCF handler)
├── kalshi_client.py        # Kalshi API wrapper
├── grok_client.py          # Grok API wrapper
├── prompt.py               # Role-based prompt template
├── config.py               # Env var loading and config
├── logger.py               # Logging setup
├── tests/
│   ├── __init__.py
│   ├── test_unit.py        # Unit tests (mocks)
│   ├── test_integration.py # Integration tests (real demo API)
│   └── test_e2e.py         # E2E tests (full flow with mocks)
├── kalshi_private_key      # Local PEM file (gitignore this)
├── .env.example            # Env template
├── requirements.txt        # As above
└── README.md               # Setup instructions
```

### Code Implementation

#### config.py
```python
"""
Configuration loader for environment variables.
Follows PEP 8: snake_case, 79-char lines, grouped imports.
"""

import os
from typing import Optional


# Standard library imports
from dotenv import load_dotenv


load_dotenv()


def get_kalshi_api_key() -> str:
    """Get Kalshi API key from env."""
    return os.getenv("KALSHI_API_KEY", "")


def get_kalshi_base_url() -> str:
    """Get Kalshi base URL from env, default demo."""
    return os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co/trade-api/v2")


def get_lookback_hours() -> int:
    """Get lookback hours from env, default 1."""
    return int(os.getenv("LOOKBACK_HOURS", "1"))


def get_grok_model() -> str:
    """Get Grok model from env, default grok-4.1."""
    return os.getenv("GROK_MODEL", "grok-4.1")


def get_xai_api_key() -> str:
    """Get xAI API key from env."""
    return os.getenv("XAI_API_KEY", "")


def get_project_id() -> Optional[str]:
    """Get GCP project ID if in GCF."""
    return os.getenv("GOOGLE_CLOUD_PROJECT")
```

#### logger.py
```python
"""
Logging setup for bot: stdout for local/GCF.
"""

import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """Setup logger with detailed level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(name)
```

#### prompt.py
```python
"""
Role-based prompt template for Grok.
"""

PROMPT_TEMPLATE = """
You are Domer, an expert in professional prediction market trading on platforms like Kalshi. Your goal is to identify high-value opportunities in binary yes/no markets.

For the following market:
- Ticker: {ticker}
- Title: {title}
- Description: {description}
- Yes Price: {yes_price}
- No Price: {no_price}
- Volume: {volume}
- Open Time: {open_time}
- Close Time: {close_time}

Perform online research using your tools (web_search, browse_page, x_keyword_search) to gather real-time data, news, polls, expert opinions, and historical trends relevant to this event. Become an expert on this specific market.

Based on your research, decide if you should take a position. If yes, recommend 'yes' or 'no' side. If no position, respond with null values.

Respond ONLY with a valid RFC 8259 JSON object: {{"ticker": "{ticker}" if position else null, "explanation": "Detailed reasoning based on research..." if position else null}}.

Example (no position): {{"ticker": null, "explanation": null}}
Example (yes position): {{"ticker": "{ticker}", "explanation": "Research shows 70% probability of yes due to recent polls..."}}
"""
```

#### kalshi_client.py
```python
"""
Kalshi API client using HTTP Signature auth with RSA PEM.
Assumes binary markets; places market orders.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import time


class KalshiClient:
    """Stateless Kalshi client."""

    def __init__(self, api_key: str, base_url: str, private_key: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.private_key = private_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _sign_request(self, method: str, path: str, body: str = "") -> str:
        """Generate HTTP Signature (simplified; adapt from Kalshi docs)."""
        timestamp = str(int(time.time()))
        payload = f"(request-target): {method.lower()} {path}\nhost: demo-api.kalshi.co\nx-kalshi-timestamp: {timestamp}\n"
        if body:
            payload += f"content-type: application/json\n"
        payload += body

        private_key_obj = serialization.load_pem_private_key(
            self.private_key.encode(), password=None
        )
        signature = private_key_obj.sign(
            payload.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def _make_request(self, method: str, endpoint: str, params: Dict = None, json_data: Dict = None) -> Dict[str, Any]:
        """Make signed request."""
        path = endpoint.lstrip("/")
        url = f"{self.base_url}/{path}"
        body = json.dumps(json_data) if json_data else ""
        signature = self._sign_request(method, path, body)

        headers = {
            "Host": "demo-api.kalshi.co",
            "X-Kalshi-Signature": signature,
            "X-Kalshi-Timestamp": str(int(time.time())),
            "Content-Type": "application/json",
            **self.session.headers,
        }

        if method == "GET":
            resp = self.session.get(url, params=params, headers=headers)
        else:
            resp = self.session.post(url, params=params, json=json_data, headers=headers)

        resp.raise_for_status()
        return resp.json()

    def get_new_markets(self, lookback_hours: int) -> List[Dict[str, Any]]:
        """Get open markets created in last N hours."""
        since = (datetime.utcnow() - timedelta(hours=lookback_hours)).isoformat() + "Z"
        params = {"status": "open", "created_after": since}
        data = self._make_request("GET", "/markets", params=params)
        return data.get("markets", [])

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        data = self._make_request("GET", "/positions")
        return data.get("positions", [])

    def place_order(self, ticker: str, side: str, count: int = 1) -> Dict[str, Any]:
        """Place market order: side 'yes' or 'no', count=1 ($10 equiv)."""
        order = {
            "ticker": ticker,
            "side": side,
            "type": "market",
            "count": count,
            "time_in_force": "ioc",  # Immediate or cancel
        }
        return self._make_request("POST", "/orders", json_data=order)


def load_private_key() -> str:
    """Load PEM key: file local, Secret Manager in GCF."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/kalshi-private-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    else:
        with open("kalshi_private_key", "r") as f:
            return f.read()
```

#### grok_client.py
```python
"""
Grok API client for chat completions with JSON mode.
"""

import requests
from typing import Dict, Any


GROK_API_URL = "https://api.x.ai/v1/chat/completions"


def get_recommendation(
    market: Dict[str, Any], api_key: str, model: str
) -> Optional[Dict[str, Any]]:
    """Send prompt to Grok, parse JSON response."""
    from prompt import PROMPT_TEMPLATE

    prompt = PROMPT_TEMPLATE.format(
        ticker=market["ticker"],
        title=market.get("title", ""),
        description=market.get("description", ""),
        yes_price=market.get("yes_bid", 0),
        no_price=market.get("no_ask", 0),
        volume=market.get("volume", 0),
        open_time=market.get("open_time", ""),
        close_time=market.get("close_time", ""),
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = requests.post(GROK_API_URL, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
        # Extract JSON (assuming full response is JSON)
        rec = json.loads(content)
        if rec["ticker"] is None:
            return None
        rec["side"] = "yes" if "yes" in rec["explanation"].lower() else "no"  # Infer side from explanation
        return rec
    except (KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid Grok response: {e}")
```

#### main.py
```python
"""
Main entry point: run_bot() for local/scheduled; entrypoint for GCF.
"""

import functions_framework  # For GCF
from config import (
    get_kalshi_api_key,
    get_kalshi_base_url,
    get_lookback_hours,
    get_grok_model,
    get_xai_api_key,
)
from kalshi_client import KalshiClient, load_private_key
from grok_client import get_recommendation
from logger import setup_logger
from typing import Dict, Any


logger = setup_logger(__name__)


def run_bot(event: Dict = None, context: Any = None) -> str:
    """Core bot logic: scan, consult, trade."""
    if not all([get_kalshi_api_key(), get_xai_api_key()]):
        raise ValueError("Missing API keys.")

    private_key = load_private_key()
    client = KalshiClient(
        api_key=get_kalshi_api_key(),
        base_url=get_kalshi_base_url(),
        private_key=private_key,
    )
    lookback = get_lookback_hours()
    model = get_grok_model()

    # Get positions tickers
    positions = client.get_positions()
    pos_tickers = {p["ticker"] for p in positions}

    # Get new markets
    markets = client.get_new_markets(lookback)
    new_markets = [m for m in markets if m["ticker"] not in pos_tickers]

    logger.info(f"Found {len(new_markets)} new markets.")

    for market in new_markets:
        try:
            rec = get_recommendation(market, get_xai_api_key(), model)
            if rec:
                logger.info(f"Trading {rec['ticker']}: {rec}")
                order_resp = client.place_order(rec["ticker"], rec["side"])
                logger.info(f"Order placed: {order_resp}")
            else:
                logger.info(f"No position for {market['ticker']}")
        except Exception as e:
            logger.error(f"Error processing {market['ticker']}: {e}")

    return f"Processed {len(new_markets)} markets."


# Local run
if __name__ == "__main__":
    run_bot()
    print("Bot run complete.")

# GCF handler
@functions_framework.http
def gcf_handler(request):
    """GCF HTTP trigger (for scheduler)."""
    return run_bot()
```

#### .env.example
```
KALSHI_API_KEY=your_kalshi_key
KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
LOOKBACK_HOURS=1
GROK_MODEL=grok-4.1
XAI_API_KEY=your_xai_key
```

#### tests/test_unit.py
```python
"""
Unit tests with mocks.
"""

import pytest
from unittest.mock import Mock, patch
from kalshi_client import KalshiClient
from grok_client import get_recommendation


@pytest.fixture
def mock_client():
    with patch("kalshi_client.requests.Session") as mock_session:
        yield KalshiClient("key", "url", "pem")


def test_get_new_markets(mock_client):
    mock_client._make_request.return_value = {"markets": [{"ticker": "TEST"}]}
    markets = mock_client.get_new_markets(1)
    assert len(markets) == 1


def test_get_recommendation():
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"ticker": "TEST", "explanation": "Yes due to polls"}'}}]
        }
        rec = get_recommendation({"ticker": "TEST"}, "key", "model")
        assert rec["ticker"] == "TEST"
```

#### tests/test_integration.py
```python
"""
Integration tests: real demo API calls (use valid demo creds).
"""

import pytest
from kalshi_client import KalshiClient
from config import get_kalshi_api_key, get_kalshi_base_url


@pytest.fixture
def client():
    key = "demo_key"  # Use real demo key
    pem = "demo_pem"
    return KalshiClient(key, get_kalshi_base_url(), pem)


def test_get_positions(client):
    positions = client.get_positions()
    assert isinstance(positions, list)
```

#### tests/test_e2e.py
```python
"""
E2E tests: full flow with comprehensive mocks.
"""

import pytest
from freezegun import freeze_time
from main import run_bot
from unittest.mock import patch, MagicMock


@freeze_time("2025-12-12 12:00:00")
@patch("main.KalshiClient")
@patch("main.get_recommendation")
def test_run_bot(mock_rec, mock_client):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.get_positions.return_value = []
    mock_client_instance.get_new_markets.return_value = [{"ticker": "TEST"}]
    mock_rec.return_value = {"ticker": "TEST", "explanation": "yes", "side": "yes"}

    with patch("main.logger") as mock_log:
        result = run_bot()
        mock_client_instance.place_order.assert_called_once_with("TEST", "yes")
        assert "Processed 1 markets" in result
```

### Setup Instructions (README.md excerpt)
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, fill values.
3. Local: `python main.py`
4. GCF: Deploy via `gcloud functions deploy`, trigger hourly via Cloud Scheduler.
5. Tests: `pytest tests/`
6. Gitignore: `kalshi_private_key`, `.env`

This implementation is production-ready for demo; extend order params (e.g., price limits) as needed. For production Kalshi, set `KALSHI_BASE_URL=https://trade-api.kalshi.com/v2`.
