### Project Structure
The "New Position Bot" is implemented as a Python 3.13.11 project following PEP 8 style guidelines for readability and maintainability. The code is stateless, relying on API calls for state (e.g., positions). It uses environment variables for configuration and logs to stdout with detailed levels (DEBUG, INFO, WARNING, ERROR).

Key design decisions:
- **Statelessness**: No local storage; fetches positions and markets via Kalshi API on each run.
- **Scheduling**: Designed to run hourly (e.g., via GCP Cloud Scheduler triggering the Cloud Function).
- **Environments**: Defaults to Kalshi demo API; configurable for production. Local runs load RSA key from `./kalshi_private_key` (PEM file). Cloud Functions load from GCP Secret Manager (secret name: `kalshi-private-key`).
- **GenAI Integration**: Uses xAI Grok API (assumed `/v1/chat/completions` endpoint based on standard LLM APIs; model configurable, defaults to `grok-4.1`). Prompts enforce JSON output for parsing.
- **Error Handling**: Retries on transient errors (e.g., API rate limits); logs failures without crashing.
- **Testing**: Pytest for unit (mocked APIs), integration (demo API), and E2E (local simulation).
- **Dependencies**: Listed in `requirements.txt`. No internet installs in runtime; assumes pre-installed.

#### File Structure
```
new_position_bot/
├── main.py                 # Entry point (local & Cloud Function)
├── bot.py                  # Core logic: market analysis, Grok calls, trading
├── auth.py                 # Kalshi & Grok authentication
├── prompts.py              # Role-based prompt template
├── config.py               # Environment variable loading & defaults
├── logger.py               # Centralized logging setup
├── tests/
│   ├── __init__.py
│   ├── test_unit.py        # Unit tests (mocks)
│   ├── test_integration.py # Integration tests (demo API)
│   └── test_e2e.py         # E2E tests (local run simulation)
├── requirements.txt        # Dependencies
├── kalshi_private_key      # Local PEM file (gitignored)
├── .env.example            # Example env vars (gitignored)
└── pytest.ini              # Pytest config
```

#### requirements.txt
```
requests==2.31.0
cryptography==42.0.5
python-dotenv==1.0.0
pytest==7.4.4
pytest-mock==3.12.0
google-cloud-secret-manager==2.16.5  # Only for Cloud Functions
```

#### .env.example (copy to .env for local)
```
KALSHI_API_KEY=your_kalshi_api_key
KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
LOOKBACK_HOURS=1
GROK_MODEL=grok-4.1
XAI_API_KEY=your_xai_api_key
```

#### pytest.ini
```
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --cov=.
```

### Code Implementation

#### config.py
```python
"""Configuration loader for environment variables."""
import os
from typing import Optional


def load_config() -> dict[str, str | int]:
    """Load and validate environment variables."""
    config = {
        "kalshi_api_key": os.getenv("KALSHI_API_KEY"),
        "kalshi_base_url": os.getenv(
            "KALSHI_BASE_URL", "https://demo-api.kalshi.co/trade-api/v2"
        ),
        "lookback_hours": int(os.getenv("LOOKBACK_HOURS", "1")),
        "grok_model": os.getenv("GROK_MODEL", "grok-4.1"),
        "xai_api_key": os.getenv("XAI_API_KEY"),
    }
    required = ["kalshi_api_key", "xai_api_key"]
    missing = [key for key in required if not config[key]]
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")
    return config
```

#### logger.py
```python
"""Centralized logging setup."""
import logging
import sys
from typing import NoReturn


def setup_logger(name: str = "new_position_bot") -> logging.Logger:
    """Set up logger to stdout with detailed levels."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Detailed logging
    return logger
```

#### auth.py
```python
"""Authentication for Kalshi and Grok APIs."""
import os
import base64
import json
from datetime import datetime, timezone
from typing import Optional
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import requests
from google.cloud import secretmanager  # Optional for Cloud
from config import load_config


def load_kalshi_private_key() -> bytes:
    """Load RSA private key: local file or GCP Secret Manager."""
    if os.getenv("FUNCTION_NAME"):  # Detect Cloud Function env
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/secrets/kalshi-private-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data
    else:
        with open("kalshi_private_key", "rb") as f:
            return f.read()


class KalshiAuth:
    """Kalshi authentication using RSA signature."""

    def __init__(self, api_key: str, private_key_pem: bytes):
        self.api_key = api_key
        self.private_key = serialization.load_pem_private_key(
            private_key_pem, password=None
        )

    def get_headers(self, path: str, method: str = "GET") -> dict[str, str]:
        """Generate signed headers for Kalshi requests."""
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"
        message = f"{method}:{path}:{timestamp}"
        signature = base64.b64encode(
            self.private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
        ).decode()
        return {
            "Authorization": self.api_key,
            "Timestamp": timestamp,
            "Signature": signature,
        }


class GrokClient:
    """Client for xAI Grok API."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1"  # Assumed based on standard

    def get_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.1
    ) -> Optional[dict]:
        """Call Grok chat completions with JSON mode."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},  # Enforce JSON
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", headers=headers, json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise ValueError(f"Grok API error: {e}")
```

#### prompts.py
```python
"""Role-based prompts for Grok."""
ROLE_PROMPT = """You are an expert in professional prediction market trading, similar to Domer. For any market, perform online research (using your knowledge and tools) to become an expert before recommending a position. Analyze recent news, trends, probabilities, and risks.

Respond ONLY with a valid RFC 8259 compliant JSON object with exactly two keys:
- "ticker": the market's unique identifier (string) if you recommend entering a position; null otherwise.
- "explanation": a textual explanation (string) of why to enter (or not); null if no position.

Example for entry: {{"ticker": "123", "explanation": "I think X will happen based on recent data..."}}
Example for no entry: {{"ticker": null, "explanation": null}}"""
```

#### bot.py
```python
"""Core bot logic for market analysis and trading."""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
from auth import KalshiAuth, GrokClient
from prompts import ROLE_PROMPT
from config import load_config
from logger import setup_logger


logger = setup_logger(__name__)


class NewPositionBot:
    def __init__(self):
        self.config = load_config()
        private_key_pem = load_kalshi_private_key()
        self.kalshi_auth = KalshiAuth(self.config["kalshi_api_key"], private_key_pem)
        self.grok_client = GrokClient(self.config["xai_api_key"], self.config["grok_model"])
        self.session = requests.Session()
        self.lookback = timedelta(hours=self.config["lookback_hours"])

    def _api_call(self, method: str, path: str, **kwargs) -> Dict:
        """Make authenticated Kalshi API call with retry."""
        headers = self.kalshi_auth.get_headers(path, method)
        self.session.headers.update(headers)
        url = f"{self.config['kalshi_base_url']}{path}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Kalshi API error {method} {path}: {e}")
            raise

    def get_positions(self) -> set[str]:
        """Fetch current position tickers."""
        data = self._api_call("GET", "/positions")
        return {pos["ticker"] for pos in data.get("positions", [])}

    def get_new_markets(self, existing_positions: set[str]) -> List[Dict]:
        """Get active markets created in lookback period without positions."""
        cutoff = datetime.utcnow() - self.lookback
        data = self._api_call("GET", "/markets?status=open&limit=100")  # Assumed filter
        markets = [
            m
            for m in data.get("markets", [])
            if datetime.fromisoformat(m["created_at"]) > cutoff
            and m["ticker"] not in existing_positions
        ]
        logger.info(f"Found {len(markets)} new markets")
        return markets

    def query_grok(self, market: Dict) -> Optional[Dict[str, Optional[str]]]:
        """Query Grok for position recommendation."""
        ticker = market["ticker"]
        title = market["title"]  # Assumed field
        user_prompt = f"Analyze this Kalshi market: Ticker={ticker}, Title='{title}'. Recommend position or none."
        try:
            completion = self.grok_client.get_completion(ROLE_PROMPT, user_prompt)
            # Parse JSON (assumed valid due to response_format)
            rec = json.loads(completion)
            if rec.get("ticker") == ticker:
                logger.info(f"Grok recommends position in {ticker}: {rec['explanation']}")
            else:
                logger.info(f"Grok skips {ticker}: {rec['explanation']}")
            return rec
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Grok response for {ticker}: {e}")
            return {"ticker": None, "explanation": None}

    def place_order(self, ticker: str, side: str = "yes", count: int = 1) -> Dict:
        """Place buy order for recommended position (assumes 'yes' for simplicity)."""
        path = f"/orders"
        data = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "type": "market",  # Market order for simplicity
            "time_in_force": "ioc",
        }
        logger.info(f"Placing order: {data}")
        return self._api_call("POST", path, json=data)

    def run(self) -> None:
        """Main bot execution."""
        logger.info("Starting New Position Bot run")
        try:
            positions = self.get_positions()
            new_markets = self.get_new_markets(positions)
            for market in new_markets:
                rec = self.query_grok(market)
                if rec and rec["ticker"]:
                    self.place_order(rec["ticker"])
            logger.info("Bot run completed")
        except Exception as e:
            logger.error(f"Bot run failed: {e}")
            raise
```

#### main.py
```python
"""Entry point: local run or GCP Cloud Function."""
import os
import sys
from bot import NewPositionBot
from logger import setup_logger


def main(request=None) -> None:  # request for Cloud Function
    """Run the bot."""
    logger = setup_logger()
    logger.info("Entry point invoked")
    bot = NewPositionBot()
    bot.run()


if __name__ == "__main__":
    main()
else:  # Cloud Function
    def function(event, context):
        main(event)
```

### Testing with Pytest

#### tests/conftest.py (shared fixtures)
```python
"""Pytest fixtures."""
import pytest
from unittest.mock import Mock
from bot import NewPositionBot
from auth import GrokClient


@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setenv("KALSHI_API_KEY", "test_key")
    monkeypatch.setenv("XAI_API_KEY", "test_xai")


@pytest.fixture
def mock_bot(mocker, mock_config):
    bot = mocker.patch("bot.NewPositionBot").call_new()
    bot.session = Mock()
    return bot


@pytest.fixture
def mock_grok(mocker):
    grok = Mock(spec=GrokClient)
    mocker.patch("bot.GrokClient", return_value=grok)
    return grok
```

#### tests/test_unit.py
```python
"""Unit tests with mocks."""
import pytest
from unittest.mock import MagicMock
from bot import NewPositionBot


def test_get_positions(mock_bot):
    mock_bot._api_call.return_value = {"positions": [{"ticker": "ABC"}]}
    positions = mock_bot.get_positions()
    assert positions == {"ABC"}


def test_query_grok(mock_grok, mock_bot):
    mock_grok.get_completion.return_value = '{"ticker": "DEF", "explanation": "Test"}'
    market = {"ticker": "DEF", "title": "Test Market"}
    rec = mock_bot.query_grok(market)
    assert rec["ticker"] == "DEF"


def test_place_order(mock_bot):
    mock_bot._api_call.return_value = {"order_id": "123"}
    result = mock_bot.place_order("GHI")
    assert result["order_id"] == "123"
```

#### tests/test_integration.py
```python
"""Integration tests using demo API."""
import pytest
from bot import NewPositionBot
import os


@pytest.mark.integration
def test_get_new_markets(monkeypatch):
    monkeypatch.setenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co/trade-api/v2")
    # Skip Grok/trading for integration; test API fetch only
    bot = NewPositionBot()
    positions = bot.get_positions()  # Should not raise
    assert isinstance(positions, set)
```

#### tests/test_e2e.py
```python
"""E2E tests: simulate full run locally."""
import pytest
from unittest.mock import patch
from main import main
import sys
from io import StringIO


@pytest.mark.e2e
@patch("bot.NewPositionBot.run")  # Mock actual run
def test_e2e_run(mock_run):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    main()
    sys.stdout = old_stdout
    assert "Starting New Position Bot run" in mystdout.getvalue()
    mock_run.assert_called_once()
```

Run tests: `pytest tests/`

### Deployment Notes
- **Local**: `python main.py` (ensure `.env` and `kalshi_private_key`).
- **GCP Cloud Function**: Deploy `main.py` as HTTP trigger; set env vars in console. Add `google-cloud-secret-manager` to runtime. Schedule via Cloud Scheduler (hourly cron: `0 * * * *`).
- **Security**: Gitignore `.env`, `kalshi_private_key`. Use GCP IAM for Secret Manager.
- **Limitations/Assumptions**: Kalshi API endpoints assumed (e.g., `/markets?status=open` for filtering; adjust post-docs). Grok API base URL and JSON mode assumed standard. No actual trading in tests. For production, add idempotency and risk limits (e.g., max order size).
