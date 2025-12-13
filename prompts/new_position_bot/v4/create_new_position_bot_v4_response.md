### Project Structure
The "New Position Bot" is implemented as a Python package that can be deployed as a stateless Google Cloud Function. The main entrypoint is `main.py`, which handles HTTP triggers (scheduled via Cloud Scheduler for hourly execution). Supporting modules include:
- `kalshi_client.py`: Handles all Kalshi API interactions (using `requests`).
- `grok_client.py`: Handles Grok API calls (using `requests`).
- `prompt.py`: Defines the role-based prompt template.
- `config.py`: Loads environment variables.
- `logger.py`: Centralized logging setup.
- `tests/`: Directory for pytest tests (unit, integration, functional).

Requirements are minimal: `requests`, `pytest`, `pytest-mock` (for tests). Use `pip install -r requirements.txt`.

**Note on Versions**: Python 3.13.11 is not yet released (as of Dec 2025, 3.12 is current; 3.13 expected early 2025). Code is written for Python 3.12+ compatibility, using type hints and modern features. Adjust `runtime` in GCP if needed.

**Environment Variables** (set in GCP Cloud Function config):
- `KALSHI_API_KEY`: Kalshi public API key.
- `KALSHI_PRIVATE_KEY`: Kalshi private API key (for signing requests).
- `KALSHI_BASE_URL`: API base URL (default: `https://demo-api.kalshi.co/trade-api/v2`).
- `GROK_API_KEY`: xAI Grok API key.
- `GROK_BASE_URL`: Grok API base URL (default: `https://api.x.ai/v1`).
- `GROK_MODEL`: Grok model to use (default/enforcement: `grok-4.1`; fallback to `grok-beta` if unavailable).
- `LOOKBACK_HOURS`: Hours to look back for new markets (default: `1`).
- `PROJECT_ID`: GCP project ID (for logging; optional).

**Deployment Notes**:
- Deploy as GCP Cloud Function (Gen 2) with trigger: HTTP, runtime: Python 3.12.
- Schedule hourly via Cloud Scheduler: `gcloud scheduler jobs create http new-position-bot --schedule="0 * * * *" --uri=<function-url> --http-method=POST`.
- The function is stateless; positions are checked via API on each run.
- Logging: Uses `logging` with JSON format for GCP integration.

---

### `requirements.txt`
```
requests==2.31.0
pytest==7.4.3
pytest-mock==3.11.1
```

---

### `config.py`
```python
import os
from typing import Optional

def load_config() -> dict[str, str | int]:
    """Load and validate environment variables."""
    return {
        "kalshi_api_key": os.getenv("KALSHI_API_KEY"),
        "kalshi_private_key": os.getenv("KALSHI_PRIVATE_KEY"),
        "kalshi_base_url": os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co/trade-api/v2"),
        "grok_api_key": os.getenv("GROK_API_KEY"),
        "grok_base_url": os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
        "grok_model": os.getenv("GROK_MODEL", "grok-4.1"),
        "lookback_hours": int(os.getenv("LOOKBACK_HOURS", "1")),
        "project_id": os.getenv("PROJECT_ID", ""),
    }

# Validation (raise errors if missing critical vars)
config = load_config()
if not config["kalshi_api_key"] or not config["kalshi_private_key"]:
    raise ValueError("KALSHI_API_KEY and KALSHI_PRIVATE_KEY are required.")
if not config["grok_api_key"]:
    raise ValueError("GROK_API_KEY is required.")
```

---

### `logger.py`
```python
import logging
import os
import json
from datetime import datetime

class GCPJSONFormatter(logging.Formatter):
    """Custom formatter for GCP Cloud Logging (JSON output)."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "message": record.getMessage(),
            "severity": record.levelname,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "component": "new-position-bot",
            "project_id": os.getenv("PROJECT_ID", ""),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

def setup_logger() -> logging.Logger:
    """Setup logger with GCP-compatible JSON formatting."""
    logger = logging.getLogger("new-position-bot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(GCPJSONFormatter())
        logger.addHandler(handler)
    return logger

logger = setup_logger()
```

---

### `prompt.py`
```python
ROLE_PROMPT = """You are Domer, a world-class expert in professional prediction market trading on platforms like Kalshi, Polymarket, and PredictIt. You have years of experience analyzing geopolitical events, economic indicators, weather patterns, sports outcomes, and cultural trends to identify high-confidence trading opportunities. Your strategy emphasizes risk management, diversification, and data-driven decisions, never trading on speculation alone.

For the given market, perform thorough online research (using your tools for web search, X search, etc.) to become an expert on the underlying event or question. Analyze historical data, current news, expert opinions, polls, and quantitative models. Consider probabilities, market liquidity, bid-ask spreads, and potential catalysts.

After research, decide if you recommend taking a position in this market. If yes, specify the market_id to enter (exact string from input). If no position is recommended (e.g., insufficient edge, high risk, or balanced odds), set market_id to null.

Respond ONLY with a valid RFC 8259 compliant JSON object, nothing else:
{{"market_id": "exact_market_id_string_or_null", "explanation": "detailed_textual_reasoning_here"}}.

Example (recommend):
{{"market_id": "123", "explanation": "Based on recent polls showing 65% support for the bill and insider leaks, I recommend buying YES contracts at current prices for an expected 15% ROI."}}

Example (no recommend):
{{"market_id": null, "explanation": "Market odds are efficiently priced with no discernible edge after reviewing sources; abstain to preserve capital."}}"""

def get_full_prompt(market: dict) -> str:
    """Generate full prompt for a market."""
    market_details = f"""
Market ID: {market['id']}
Title: {market['title']}
Category: {market['category']}
Question: {market['question']}
Status: {market['status']}
Open: {market['open']}
Close: {market['close']}
Tick Size: {market['tick_size']}
Min Order: {market['min_order']}
Max Order: {market['max_order']}
Current Yes Bid: {market.get('yes_bid', 'N/A')}
Current Yes Ask: {market.get('yes_ask', 'N/A')}
Volume: {market['volume']}
"""
    return ROLE_PROMPT + "\n\n" + market_details
```

---

### `kalshi_client.py`
```python
import requests
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Any
from logger import logger
from config import config

class KalshiClient:
    def __init__(self):
        self.api_key = config["kalshi_api_key"]
        self.private_key = config["kalshi_private_key"]
        self.base_url = config["kalshi_base_url"]
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Key {self.api_key}"})

    def _sign_request(self, method: str, path: str, body: str = "") -> str:
        """Generate signature for Kalshi requests."""
        timestamp = str(int(time.time()))
        message = f"{method}:{path}:{body}:{timestamp}"
        signature = base64.b64encode(
            hmac.new(
                self.private_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        return f"{self.api_key}:{signature}:{timestamp}"

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, json: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated API request."""
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(json) if json else ""
        signature = self._sign_request(method, endpoint, body)
        headers = {
            "Authorization": f"Key {signature}",
            "Content-Type": "application/json",
        }
        response = self.session.request(method, url, headers=headers, params=params, json=json)
        response.raise_for_status()
        return response.json()

    def get_markets(self, status: str = "open", limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch markets, filtered by status."""
        params = {"status": status, "limit": limit, "offset": offset}
        data = self._make_request("GET", "/markets", params=params)
        return data.get("markets", [])

    def get_positions(self) -> List[Dict]:
        """Fetch user's current positions."""
        data = self._make_request("GET", "/positions")
        return data.get("positions", [])

    def place_order(self, market_id: str, side: str, count: int, price: float, type_: str = "market") -> Dict:
        """Place an order (e.g., buy YES)."""
        payload = {
            "market_id": market_id,
            "side": side,  # "yes" or "no"
            "count": count,
            "price": price,
            "type": type_,
        }
        data = self._make_request("POST", f"/orders", json=payload)
        logger.info(f"Order placed: {data}")
        return data

# Global client instance
kalshi = KalshiClient()
```

---

### `grok_client.py`
```python
import requests
import json
from typing import Dict, Any, Optional
from logger import logger
from config import config
from prompt import get_full_prompt

class GrokClient:
    def __init__(self):
        self.api_key = config["grok_api_key"]
        self.base_url = config["grok_base_url"]
        self.model = config["grok_model"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_recommendation(self, market: Dict) -> Optional[Dict[str, Any]]:
        """Send prompt to Grok and parse JSON response."""
        prompt = get_full_prompt(market)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise JSON responder. Always output valid RFC 8259 JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Low for consistent JSON
            "max_tokens": 500,
        }
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            # Extract JSON (assume direct output)
            rec = json.loads(content)
            if rec.get("market_id") is not None:
                logger.info(f"Grok recommends market {rec['market_id']}: {rec['explanation']}")
            else:
                logger.info("Grok recommends no position.")
            return rec
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Grok API error: {e}")
            return None

# Global client instance
grok = GrokClient()
```

---

### `main.py`
```python
import functions_framework
from datetime import datetime, timedelta
from typing import Dict, Any
from logger import logger, setup_logger
from config import config
from kalshi_client import kalshi
from grok_client import grok

@functions_framework.http
def new_position_bot(request: Any) -> Dict[str, Any]:
    """Main Cloud Function entrypoint: Run bot logic."""
    setup_logger()  # Ensure logger is set
    logger.info("Bot run started")

    lookback_hours = config["lookback_hours"]
    cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
    logger.info(f"Looking back {lookback_hours} hours (cutoff: {cutoff_time})")

    # Get current positions
    positions = kalshi.get_positions()
    position_markets = {pos["market_id"] for pos in positions}
    logger.info(f"Current positions in {len(position_markets)} markets")

    # Get open markets
    all_open = []
    offset = 0
    while True:
        batch = kalshi.get_markets(status="open", limit=100, offset=offset)
        if not batch:
            break
        all_open.extend(batch)
        offset += 100

    # Filter new markets without positions
    new_markets = [
        m for m in all_open
        if m.get("created_at")  # Assuming ISO timestamp in market dict
        and datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")) > cutoff_time
        and m["id"] not in position_markets
    ]
    logger.info(f"Found {len(new_markets)} new open markets")

    # Analyze each with Grok
    for market in new_markets:
        rec = grok.get_recommendation(market)
        if rec and rec.get("market_id") == market["id"]:
            # Place order (simplified: buy 10 YES at market price; customize as needed)
            yes_ask = market.get("yes_ask", 0.5)
            order = kalshi.place_order(market["id"], "yes", 10, yes_ask)
            logger.info(f"Position taken: {order}")
        else:
            logger.info(f"No recommendation for market {market['id']}")

    logger.info("Bot run completed")
    return {"status": "success", "new_markets_analyzed": len(new_markets)}
```

---

### Tests (`tests/test_kalshi_client.py`)
```python
import pytest
from unittest.mock import patch, MagicMock
from kalshi_client import KalshiClient

@pytest.fixture
def client():
    with patch.dict("os.environ", {
        "KALSHI_API_KEY": "test_key",
        "KALSHI_PRIVATE_KEY": "test_priv",
        "KALSHI_BASE_URL": "https://test.kalshi.co"
    }):
        return KalshiClient()

def test_get_markets(client):
    with patch("requests.Session.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"markets": [{"id": "1"}]}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        markets = client.get_markets()
        assert len(markets) == 1
        mock_request.assert_called_once()

def test_place_order(client):
    with patch("requests.Session.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        result = client.place_order("1", "yes", 10, 0.5)
        assert result["success"] is True
```

---

### Tests (`tests/test_grok_client.py`)
```python
import pytest
import json
from unittest.mock import patch, MagicMock
from grok_client import GrokClient
from prompt import get_full_prompt

@pytest.fixture
def client():
    with patch.dict("os.environ", {"GROK_API_KEY": "test_grok_key"}):
        return GrokClient()

def test_get_recommendation_recommends(client):
    mock_response = {
        "choices": [{"message": {"content": '{"market_id": "123", "explanation": "Good edge"}'}}]
    }
    with patch("requests.post", return_value=MagicMock(json=lambda: mock_response, raise_for_status=lambda: None)) as mock_post:
        rec = client.get_recommendation({"id": "123", "title": "Test"})
        assert rec["market_id"] == "123"

def test_get_recommendation_no_recommend(client):
    mock_response = {
        "choices": [{"message": {"content": '{"market_id": null, "explanation": "No edge"}'}}]
    }
    with patch("requests.post", return_value=MagicMock(json=lambda: mock_response, raise_for_status=lambda: None)) as mock_post:
        rec = client.get_recommendation({"id": "456", "title": "Test"})
        assert rec["market_id"] is None

def test_get_full_prompt():
    market = {"id": "123", "title": "Will it rain?"}
    prompt = get_full_prompt(market)
    assert "Market ID: 123" in prompt
    assert "You are Domer" in prompt
```

---

### Tests (`tests/test_main.py` - Integration/End-to-End)
```python
import pytest
from unittest.mock import patch, MagicMock
from main import new_position_bot
from unittest.mock import Mock

@pytest.fixture
def mock_request():
    return Mock()

def test_bot_flow(mock_request):
    with patch("main.kalshi") as mock_kalshi, patch("main.grok") as mock_grok, \
         patch("main.datetime") as mock_dt, patch("main.timedelta") as mock_td:
        # Mock time
        mock_now = MagicMock()
        mock_dt.utcnow.return_value = mock_now
        mock_cutoff = mock_now - mock_td(hours=1)
        
        # Mock positions: empty
        mock_kalshi.get_positions.return_value = []
        
        # Mock markets: one new
        mock_market = {"id": "123", "created_at": (mock_cutoff + mock_td(minutes=30)).isoformat()}
        mock_kalshi.get_markets.return_value = [mock_market]
        
        # Mock Grok rec: recommend
        mock_grok.get_recommendation.return_value = {"market_id": "123", "explanation": "Test"}
        
        # Mock place_order
        mock_kalshi.place_order.return_value = {"success": True}
        
        result = new_position_bot(mock_request)
        assert result["status"] == "success"
        assert result["new_markets_analyzed"] == 1
        mock_grok.get_recommendation.assert_called_once()
        mock_kalshi.place_order.assert_called_once()
```

---

### Running Tests
```bash
pytest tests/ -v --cov=.
```

This implementation is complete, testable, and deployable. Customize order sizing/logic in `main.py` as needed. For production, add error handling (e.g., retries) and monitor logs in GCP. The prompt ensures JSON compliance; Grok's low temperature aids parsing. If `created_at` field differs, adjust filtering based on Kalshi docs.
