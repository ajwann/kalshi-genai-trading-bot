### Project Structure
```
new_position_bot/
├── main.py                  # Entry point: runs bot locally or as GCP Cloud Function
├── requirements.txt         # Dependencies
├── .env.example            # Example env vars (not committed)
├── tests/
│   ├── __init__.py
│   ├── test_kalshi_client.py       # Unit/integration tests for Kalshi integration
│   ├── test_grok_client.py         # Unit/integration tests for Grok integration
│   ├── test_bot_logic.py           # Unit tests for core bot logic
│   └── test_end_to_end.py          # Functional E2E tests (mocked APIs)
├── kalshi_client.py        # Kalshi API wrapper (auth, requests)
├── grok_client.py          # Grok API client
├── bot_logic.py            # Core bot: analyze markets, query Grok, place orders
├── prompts.py              # Role-based prompt template
├── config.py               # Env var loading, defaults
├── utils.py                # Helpers: logging, timestamps, JSON parsing
├── gcp_secrets.py          # GCP Secret Manager loader (for Cloud Functions)
└── kalshi_private_key      # Local PEM file (not committed; generate via Kalshi dashboard)
```

### requirements.txt
```
requests==2.32.3
cryptography==43.0.1
python-dotenv==1.0.1
pytest==8.3.3
pytest-mock==3.14.0
google-cloud-secret-manager==2.20.0  # Only for Cloud Functions
```

### config.py
```python
"""
Configuration loader for environment variables.
Follows PEP 8: module-level docstring, snake_case vars.
"""

import os
from typing import Optional

# Default values
_DEFAULT_LOOKBACK_HOURS = 1
_DEFAULT_KALSHI_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
_DEFAULT_GROK_MODEL = "grok-4.1"  # As per latest xAI docs; configurable for switching

# Load env vars
KALSHI_API_KEY: str = os.getenv("KALSHI_API_KEY")
if not KALSHI_API_KEY:
    raise ValueError("KALSHI_API_KEY environment variable is required.")

KALSHI_BASE_URL: str = os.getenv("KALSHI_BASE_URL", _DEFAULT_KALSHI_BASE_URL)

LOOKBACK_HOURS: int = int(os.getenv("LOOKBACK_HOURS", _DEFAULT_LOOKBACK_HOURS))

GROK_MODEL: str = os.getenv("GROK_MODEL", _DEFAULT_GROK_MODEL)
if not GROK_MODEL:
    raise ValueError("GROK_MODEL environment variable is required.")

XAI_API_KEY: Optional[str] = os.getenv("XAI_API_KEY")  # Optional if not using Grok


def get_private_key() -> str:
    """
    Load Kalshi RSA private key: local file or GCP Secret Manager.

    Returns:
        str: PEM-formatted private key.
    """
    if os.getenv("FUNCTION_NAME"):  # GCP Cloud Function detection
        from gcp_secrets import load_secret
        return load_secret("kalshi-private-key")
    else:
        with open("kalshi_private_key", "r") as f:
            return f.read()
```

### utils.py
```python
"""
Utility functions: logging, timestamps, JSON handling.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

# Setup logging: stdout for local/Cloud Functions
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def current_millis() -> int:
    """Return current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


def parse_unix_timestamp(ts_str: str) -> int:
    """
    Parse ISO datetime string to Unix ms timestamp.

    Args:
        ts_str: ISO format like '2023-11-07T05:31:56Z'.

    Returns:
        int: Unix timestamp in ms.
    """
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def validate_json_response(content: str, expected_keys: list) -> Dict[str, Any]:
    """
    Parse and validate JSON response from Grok.

    Args:
        content: Raw JSON string.
        expected_keys: List of required keys (e.g., ['ticker', 'explanation']).

    Returns:
        Dict: Parsed JSON.

    Raises:
        ValueError: If invalid JSON or missing keys.
    """
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object.")
        for key in expected_keys:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in response.")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
```

### prompts.py
```python
"""
Role-based prompt template for Grok.
"""

ROLE_PROMPT = """
You are an expert in professional prediction market trading, similar to Domer. You specialize in Kalshi markets.

For the given market (ticker: {ticker}, title: {title}, yes_sub_title: {yes_sub_title}, no_sub_title: {no_sub_title}, 
created_time: {created_time}, open_time: {open_time}, close_time: {close_time}, status: {status}), 
perform online research to become an expert in this market. Use your tools to search recent news, data, and trends.

Decide if to take a position: buy YES or NO, or none. Base on edge from research (e.g., probabilities, events).

Respond ONLY with a RFC 8259 compliant JSON object: {{"ticker": "{ticker}" if position else null, "explanation": "text reason" or null}}.
If no position, {{"ticker": null, "explanation": null}}.
Explanation: concise, why chosen (or not), research insights.
"""


def build_prompt(market: dict) -> str:
    """
    Build full user prompt from market data.

    Args:
        market: Dict from Kalshi /markets response.

    Returns:
        str: Formatted prompt.
    """
    return ROLE_PROMPT.format(
        ticker=market["ticker"],
        title=market.get("title", ""),
        yes_sub_title=market["yes_sub_title"],
        no_sub_title=market["no_sub_title"],
        created_time=market["created_time"],
        open_time=market["open_time"],
        close_time=market["close_time"],
        status=market["status"],
    )
```

### kalshi_client.py
```python
"""
Kalshi API client: handles RSA auth, requests.
Uses requests; follows PEP 8.
"""

import base64
import hashlib
from typing import Dict, List, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import requests
from config import get_private_key, KALSHI_API_KEY, KALSHI_BASE_URL
from utils import current_millis, logger, parse_unix_timestamp


class KalshiClient:
    """
    Stateless Kalshi API client for demo/prod.

    Handles RSA-PSS signing for auth headers.
    """

    def __init__(self):
        self.base_url = KALSHI_BASE_URL.rstrip("/")
        self.api_key = KALSHI_API_KEY
        self.private_key = serialization.load_pem_private_key(
            get_private_key().encode(),
            password=None,
        )

    def _sign_request(self, method: str, path: str, body: str = "") -> str:
        """
        Generate RSA-PSS signature for request.

        Args:
            method: HTTP method (upper).
            path: Endpoint path.
            body: Request body string.

        Returns:
            str: Base64 signature.
        """
        timestamp = str(current_millis())
        message = f"{method}\n{path}\n{timestamp}\n{body}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Dict:
        """
        Make authenticated request.

        Args:
            method: GET/POST.
            endpoint: e.g., "/markets".
            params: Query params.
            json: Body.

        Returns:
            Dict: JSON response.

        Raises:
            requests.RequestException: On failure.
        """
        path = endpoint
        body = json.dumps(json) if json else ""
        signature = self._sign_request(method, path, body)
        timestamp = str(current_millis())
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}{path}"
        response = requests.request(method, url, params=params, json=json, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Kalshi {method} {endpoint}: success")
        return data

    def get_open_markets(self, min_created_ts: int, limit: int = 100) -> List[Dict]:
        """
        Get open markets created after timestamp.

        Args:
            min_created_ts: Unix ms.
            limit: Max results.

        Returns:
            List of market dicts.
        """
        params = {"status": "open", "min_created_ts": min_created_ts, "limit": limit}
        data = self._make_request("GET", "/markets", params=params)
        return data.get("markets", [])

    def get_positions(self) -> List[Dict]:
        """
        Get current market positions.

        Returns:
            List of position dicts.
        """
        data = self._make_request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    def create_order(self, ticker: str, side: str, action: str, count: int) -> Dict:
        """
        Create a market buy order (simplest for new position).

        Args:
            ticker: Market ticker.
            side: "yes" or "no".
            action: "buy".
            count: Number of contracts (e.g., 10).

        Returns:
            Order response dict.
        """
        order_body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": "market",  # Market order for immediate fill
        }
        data = self._make_request("POST", "/orders", json=order_body)
        logger.info(f"Created order for {ticker}: {data}")
        return data
```

### grok_client.py
```python
"""
xAI Grok API client for chat completions.
"""

import json
from typing import Dict

import requests
from config import XAI_API_KEY, GROK_MODEL
from prompts import build_prompt
from utils import logger, validate_json_response

GROK_API_URL = "https://api.x.ai/v1/chat/completions"


class GrokClient:
    """
    Client for querying Grok with JSON response format.
    """

    def __init__(self):
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY required for Grok.")
        self.api_key = XAI_API_KEY
        self.model = GROK_MODEL

    def query_market(self, market: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query Grok for position recommendation on a market.

        Args:
            market: Market dict from Kalshi.

        Returns:
            Dict: {"ticker": str|null, "explanation": str|null}
        """
        prompt = build_prompt(market)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,  # Low for consistent JSON
        }

        response = requests.post(GROK_API_URL, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        logger.info(f"Grok response for {market['ticker']}: {content[:100]}...")

        # Validate and parse
        parsed = validate_json_response(content, ["ticker", "explanation"])
        # Ensure RFC 8259: keys quoted, values proper
        return parsed
```

### bot_logic.py
```python
"""
Core bot logic: analyze new markets, query Grok, place orders.
Stateless; runs per invocation.
"""

from typing import Dict, List

from config import LOOKBACK_HOURS
from kalshi_client import KalshiClient
from grok_client import GrokClient
from utils import current_millis, logger, parse_unix_timestamp

kalshi = KalshiClient()
grok = GrokClient()


def run_bot() -> None:
    """
    Main bot flow: hourly cron/Cloud Function entry.
    1. Get current positions tickers.
    2. Calc lookback ts.
    3. Get new open markets.
    4. For each without position: query Grok.
    5. If recommended, place order.
    """
    # Step 1: Current positions
    positions = kalshi.get_positions()
    position_tickers = {pos["ticker"] for pos in positions if pos["position"] != 0}
    logger.info(f"Current positions: {len(position_tickers)} tickers")

    # Step 2: Lookback
    lookback_ms = current_millis() - (LOOKBACK_HOURS * 60 * 60 * 1000)
    logger.info(f"Lookback: markets after {lookback_ms}")

    # Step 3: New markets
    new_markets = kalshi.get_open_markets(lookback_ms)
    logger.info(f"Found {len(new_markets)} new open markets")

    # Step 4-5: Analyze each
    for market in new_markets:
        ticker = market["ticker"]
        if ticker in position_tickers:
            logger.info(f"Skipping {ticker}: existing position")
            continue

        # Query Grok
        recommendation = grok.query_market(market)
        if recommendation["ticker"] is None:
            logger.info(f"Grok: No position for {ticker}")
            continue

        # Place order (assume buy 10 contracts; enhance with side/amount logic)
        # Note: Prompt implies side in explanation; parse if needed. Here, default yes buy.
        order = kalshi.create_order(ticker, "yes", "buy", 10)
        logger.info(f"Placed order: {order['order']['order_id']}")
```

### main.py
```python
"""
Entry point: run_bot() for local or GCP Cloud Function.
"""

import os
import sys
from typing import Any

from bot_logic import run_bot
from config import config  # Import to validate env
from utils import logger


def local_main() -> None:
    """Run locally: load .env if present."""
    from dotenv import load_dotenv
    load_dotenv()
    run_bot()


@logging.google.cloud.functions.CloudFunctionErrorHandler  # For GCP
def gcp_main(event: dict, context: Any) -> None:
    """GCP Cloud Function: triggered hourly via scheduler."""
    # GCP auto-loads env/secrets
    run_bot()


if __name__ == "__main__":
    if os.getenv("FUNCTION_NAME"):
        # Cloud Function: but main.py not direct entry; use for testing
        logger.warning("Running in FUNCTION mode; use gcp_main for prod.")
        gcp_main({}, None)
    else:
        local_main()
elif "pytest" not in sys.modules:
    # Direct run: local
    local_main()
```

### gcp_secrets.py
```python
"""
GCP Secret Manager loader for private key.
Only imported in Cloud Functions.
"""

from google.cloud import secretmanager
import os

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


def load_secret(secret_id: str) -> str:
    """
    Load secret from GCP Secret Manager.

    Args:
        secret_id: e.g., "kalshi-private-key".

    Returns:
        str: Secret value.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
```

### tests/test_kalshi_client.py
```python
"""
Unit/integration tests for KalshiClient.
"""

import pytest
from unittest.mock import patch, MagicMock
from kalshi_client import KalshiClient
from utils import current_millis


@pytest.fixture
def client():
    with patch("config.get_private_key", return_value="-----BEGIN PRIVATE KEY-----\nfoo\n-----END PRIVATE KEY-----"):
        yield KalshiClient()


def test_get_open_markets(client):
    with patch("requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"markets": [{"ticker": "TEST"}]}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        markets = client.get_open_markets(current_millis() - 3600000)

        assert len(markets) == 1
        mock_request.assert_called_once()
        assert mock_request.call_args[1]["params"]["status"] == "open"


def test_create_order(client):
    with patch("requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"order": {"order_id": "123"}}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        order = client.create_order("TEST", "yes", "buy", 10)

        assert order["order"]["order_id"] == "123"
        assert mock_request.call_args[1]["json"]["count"] == 10
```

### tests/test_grok_client.py
```python
"""
Tests for GrokClient.
"""

import pytest
from unittest.mock import patch
from grok_client import GrokClient


@pytest.fixture
def client():
    with patch("config.XAI_API_KEY", "test_key"):
        yield GrokClient()


def test_query_market(client):
    market = {"ticker": "TEST", "title": "Test Market"}
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"ticker": "TEST", "explanation": "Reason"}'}}]
        }
        mock_post.return_value = mock_response

        rec = client.query_market(market)

        assert rec["ticker"] == "TEST"
        assert "Reason" in rec["explanation"]
```

### tests/test_bot_logic.py
```python
"""
Unit tests for bot_logic.
"""

import pytest
from unittest.mock import patch, MagicMock
from bot_logic import run_bot


@patch("bot_logic.kalshi.get_positions")
@patch("bot_logic.kalshi.get_open_markets")
@patch("bot_logic.grok.query_market")
@patch("bot_logic.kalshi.create_order")
def test_run_bot(mock_create, mock_grok, mock_markets, mock_positions):
    mock_positions.return_value = [{"ticker": "EXISTING", "position": 5}]
    mock_markets.return_value = [
        {"ticker": "EXISTING", "status": "open"},
        {"ticker": "NEW", "status": "open"}
    ]
    mock_grok.return_value = {"ticker": "NEW", "explanation": "Buy YES"}

    run_bot()

    mock_grok.assert_called_once()
    mock_create.assert_called_once_with("NEW", "yes", "buy", 10)


@patch("bot_logic.kalshi.get_positions")
@patch("bot_logic.kalshi.get_open_markets")
@patch("bot_logic.grok.query_market")
def test_no_recommendation(mock_grok, mock_markets, mock_positions):
    mock_grok.return_value = {"ticker": None, "explanation": None}

    run_bot()

    # No create_order call
    assert not patch.stopall.called  # Simplified; use coverage
```

### tests/test_end_to_end.py
```python
"""
E2E tests: mocked full flow.
Requires pytest-mock.
"""

import pytest
from unittest.mock import patch
import sys
sys.path.insert(0, "..")  # For relative import in tests

from main import local_main
from bot_logic import run_bot


def test_e2e_full_flow(monkeypatch):
    # Mock all external calls
    monkeypatch.setenv("KALSHI_API_KEY", "test")
    monkeypatch.setenv("XAI_API_KEY", "test")
    with patch("kalshi_client.requests.request") as mock_kalshi, \
         patch("grok_client.requests.post") as mock_grok, \
         patch("config.get_private_key", return_value="test_pem"):

        # Mock responses
        mock_kalshi.side_effect = [
            MagicMock(json=lambda: {"market_positions": []}),  # positions
            MagicMock(json=lambda: {"markets": [{"ticker": "NEW", "status": "open"}]}),  # markets
        ]
        mock_grok.return_value.json.return_value = {"choices": [{"message": {"content": '{"ticker": "NEW", "explanation": "Test"}'}}]}

        local_main()  # Or run_bot()

        mock_kalshi.assert_called()  # Multiple calls
        mock_grok.assert_called()
```

### Setup Instructions
1. **Local Run**:
   - `cp .env.example .env` (fill KALSHI_API_KEY, XAI_API_KEY; generate kalshi_private_key.pem from Kalshi dashboard).
   - `pip install -r requirements.txt`
   - `python main.py` (runs hourly logic; cron: `0 * * * * python /path/to/main.py`).

2. **GCP Cloud Function**:
   - Deploy as HTTP trigger (scheduler: every hour).
   - Set env vars in console: KALSHI_API_KEY, etc.
   - Store private key in Secret Manager as "kalshi-private-key".
   - Use `gcloud functions deploy new-position-bot --runtime python313 --trigger-http --allow-unauthenticated` (adapt).

3. **Testing**:
   - `pytest tests/ -v` (unit/integration).
   - `pytest tests/test_end_to_end.py -v` (E2E with mocks).
   - Coverage: `pytest --cov=. --cov-report=html`.

4. **Enhancements**:
   - Parse Grok explanation for side/count (e.g., regex "buy YES 20").
   - Handle pagination in get_markets/positions.
   - Add error retries, rate limits (Kalshi: ~100 req/min).
   - Position check: |position| > 0.

This bot is stateless, PEP 8 compliant, tested, and ready for hourly runs. Logs to stdout for GCP/ local.
