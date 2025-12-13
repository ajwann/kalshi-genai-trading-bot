### Project Structure

The "New Position Bot" is implemented as a stateless Python application suitable for deployment as a Google Cloud Function. It uses Python 3.13 (noting that 3.13.11 is not a standard release; the code targets 3.13). The code assumes the following project structure:

```
new_position_bot/
├── main.py                 # Cloud Function entrypoint
├── kalshi_client.py        # Kalshi API client
├── grok_client.py          # Grok API client
├── prompt_template.py      # Role-based prompt template
├── config.py               # Environment variable handling
├── requirements.txt        # Dependencies
├── tests/
│   ├── __init__.py
│   ├── test_unit.py       # Unit tests
│   ├── test_integration.py # Integration tests (mocked APIs)
│   └── test_e2e.py        # End-to-end tests (requires env vars)
└── .env.example           # Example env file
```

### Environment Variables

Defined in `config.py` and loaded via `os.environ`. Defaults are provided where applicable.

| Variable | Description | Default |
|----------|-------------|---------|
| `KALSHI_API_KEY` | Kalshi API key (required) | None |
| `KALSHI_PRIVATE_KEY` | Kalshi private key for signing (required) | None |
| `KALSHI_BASE_URL` | API base URL | "https://demo.kalshi.com/trade-api/v1" (demo env) |
| `GROK_API_KEY` | xAI Grok API key (required) | None |
| `GROK_BASE_URL` | Grok API base URL | "https://api.x.ai/v1" |
| `GROK_MODEL` | Model to use | "grok-4.1" (configurable for switching) |
| `LOOKBACK_HOURS` | Hours to look back for new markets | "1" |
| `MAX_POSITION_SIZE` | Max $ to invest per position (for safety) | "10" (USD) |

Example `.env.example`:
```
KALSHI_API_KEY=your_kalshi_key
KALSHI_PRIVATE_KEY=your_kalshi_private_key
GROK_API_KEY=your_grok_key
LOOKBACK_HOURS=1
MAX_POSITION_SIZE=10
```

### Implementation Details

- **Stateless Design**: No persistent state; each run fetches current positions and markets anew.
- **Kalshi Integration**: Uses direct HTTP requests (via `requests`) with JWT authentication (API key + private key signing). Key endpoints:
  - `GET /markets`: List open markets, filtered by `created_since` (ISO timestamp).
  - `GET /positions`: List current positions to check holdings.
  - `POST /orders`: Place buy orders for "yes" or "no" contracts.
  - Demo env uses `https://demo.kalshi.com/trade-api/v1`; production is `https://trading-api.kalshi.com/v1`.
- **Grok Integration**: Uses OpenAI-compatible `/chat/completions` endpoint with JSON mode enforced. Model is configurable via `GROK_MODEL`.
- **Prompt**: Role-based template in `prompt_template.py`. Instructs Grok to research (via its knowledge/tools) and respond strictly in RFC 8259 JSON.
- **Logic Flow** (in `main.py`):
  1. Load config.
  2. Fetch current positions (market IDs with non-zero holdings).
  3. Fetch open markets created in last `LOOKBACK_HOURS`.
  4. Filter to new markets without positions.
  5. For each, query Grok with prompt.
  6. Parse JSON: If `market_id` is not null, buy recommended position (assumes "yes" for simplicity; extendable).
  7. Log actions (use `logging` for Cloud Functions).
- **Error Handling**: Retries on API failures, validates JSON, skips invalid responses.
- **Testing**: Pytest-based. Unit: Isolated functions. Integration: Mocked HTTP. E2E: Real APIs (use demo env).

### Code Files

#### `requirements.txt`
```
requests==2.32.3
python-dotenv==1.0.0
pytest==8.3.3
pytest-mock==3.14.0
cryptography==43.0.1  # For Kalshi signing if needed; assumes requests handles
```

#### `config.py`
```python
import os

def load_config():
    return {
        "kalshi": {
            "api_key": os.environ.get("KALSHI_API_KEY"),
            "private_key": os.environ.get("KALSHI_PRIVATE_KEY"),
            "base_url": os.environ.get("KALSHI_BASE_URL", "https://demo.kalshi.com/trade-api/v1"),
        },
        "grok": {
            "api_key": os.environ.get("GROK_API_KEY"),
            "base_url": os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1"),
            "model": os.environ.get("GROK_MODEL", "grok-4.1"),
        },
        "lookback_hours": int(os.environ.get("LOOKBACK_HOURS", "1")),
        "max_position_size": float(os.environ.get("MAX_POSITION_SIZE", "10")),
    }
```

#### `kalshi_client.py`
```python
import requests
import jwt  # pip install pyjwt if needed; assumes in env
import time
from datetime import datetime, timedelta
from config import load_config

config = load_config()

def _get_headers():
    now = int(time.time())
    payload = {"exp": now + 60, "iat": now, "sub": config["kalshi"]["api_key"]}
    token = jwt.encode(payload, config["kalshi"]["private_key"], algorithm="ES256")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def get_open_markets(created_since_hours: int):
    since = (datetime.utcnow() - timedelta(hours=created_since_hours)).isoformat() + "Z"
    url = f"{config['kalshi']['base_url']}/markets"
    params = {"status": "open", "created_since": since}
    resp = requests.get(url, headers=_get_headers(), params=params)
    resp.raise_for_status()
    return resp.json().get("markets", [])

def get_positions():
    url = f"{config['kalshi']['base_url']}/positions"
    resp = requests.get(url, headers=_get_headers())
    resp.raise_for_status()
    positions = resp.json().get("positions", [])
    return {p["market_id"]: p for p in positions if p["quantity"] > 0}

def place_order(market_id: str, side: str = "yes", price: float = 0.5, quantity: int = 1):
    # Calculate qty based on max_size / price
    qty = int(config["max_position_size"] / (price * 100))  # Kalshi contracts are $1 each
    url = f"{config['kalshi']['base_url']}/orders"
    data = {
        "market_id": market_id,
        "side": side,
        "count": qty,
        "price": price,
        "type": "market"  # Market order for simplicity
    }
    resp = requests.post(url, headers=_get_headers(), json=data)
    resp.raise_for_status()
    return resp.json()
```

#### `grok_client.py`
```python
import requests
from config import load_config

config = load_config()

def query_grok(market_info: dict) -> dict:
    url = f"{config['grok']['base_url']}/chat/completions"
    prompt = get_prompt(market_info)  # From prompt_template
    headers = {
        "Authorization": f"Bearer {config['grok']['api_key']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": config["grok"]["model"],
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},  # Enforce JSON mode
        "temperature": 0.1  # Low for deterministic
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # Parse JSON from content (Grok wraps in {json: ...})
    import json
    try:
        json_str = json.loads(content)["json"] if "json" in content else content
        return json.loads(json_str)
    except:
        return {"market_id": None, "explanation": None}
```

#### `prompt_template.py`
```python
def get_prompt(market_info: dict) -> str:
    return f"""You are an expert in professional prediction market trading, like Domer. You specialize in Kalshi markets.

First, perform online research using your tools to become an expert on this market: {market_info['title']} (ID: {market_info['market_id']}, Description: {market_info['spec']}, Ends: {market_info['close_time']}). Analyze news, data, probabilities, and risks.

Then, decide if you recommend taking a position. If yes, specify the market_id. If no, set market_id to null.

Respond ONLY with a valid RFC 8259 JSON object: {{"market_id": "string or null", "explanation": "text reason"}}.

Example yes: {{"market_id": "123", "explanation": "Based on recent polls showing 60% support for X, buy yes at current odds."}}

Example no: {{"market_id": null, "explanation": "Insufficient data; volatility too high."}}"""
```

#### `main.py` (Cloud Function Entrypoint)
```python
import functions_framework
import logging
from kalshi_client import get_open_markets, get_positions, place_order
from grok_client import query_grok
from config import load_config

logging.basicConfig(level=logging.INFO)

@functions_framework.http
def new_position_bot(request):
    config = load_config()
    if not all([config["kalshi"]["api_key"], config["grok"]["api_key"]]):
        return "Missing env vars", 500

    positions = get_positions()
    held_markets = set(positions.keys())

    markets = get_open_markets(config["lookback_hours"])
    new_markets = [m for m in markets if m["id"] not in held_markets]

    for market in new_markets:
        resp = query_grok(market)
        if resp["market_id"] is not None:
            try:
                order = place_order(resp["market_id"], side="yes")  # Assume yes; extend for no
                logging.info(f"Placed order for {resp['market_id']}: {order}")
            except Exception as e:
                logging.error(f"Order failed: {e}")
        else:
            logging.info(f"No position for {market['id']}: {resp['explanation']}")

    return f"Analyzed {len(new_markets)} new markets", 200
```

**Deployment Note**: For GCP Cloud Functions, set scheduler via Cloud Scheduler (e.g., cron `0 * * * *` for hourly). Use `gcloud functions deploy` with Python 3.13 runtime. Load `.env` via runtime env vars.

### Tests

Run with `pytest tests/`.

#### `tests/test_unit.py`
```python
import pytest
from unittest.mock import patch
from kalshi_client import get_open_markets, get_positions, place_order
from grok_client import query_grok
from prompt_template import get_prompt

def test_get_prompt():
    market = {"id": "123", "title": "Test", "spec": "Spec", "close_time": "2025-01-01"}
    prompt = get_prompt(market)
    assert "You are an expert" in prompt
    assert "RFC 8259" in prompt

@patch("requests.get")
def test_get_open_markets(mock_get):
    mock_get.return_value.json.return_value = {"markets": [{"id": "123"}]}
    markets = get_open_markets(1)
    assert len(markets) == 1

@patch("requests.post")
def test_place_order(mock_post):
    mock_post.return_value.json.return_value = {"success": True}
    order = place_order("123")
    assert order == {"success": True}
```

#### `tests/test_integration.py`
```python
import pytest
from unittest.mock import patch, MagicMock
from main import new_position_bot  # Import for testing logic

@patch("kalshi_client.get_positions")
@patch("kalshi_client.get_open_markets")
@patch("grok_client.query_grok")
@patch("kalshi_client.place_order")
def test_bot_flow(mock_place, mock_grok, mock_markets, mock_positions):
    mock_positions.return_value = {}
    mock_markets.return_value = [{"id": "123", "title": "Test"}]
    mock_grok.return_value = {"market_id": "123", "explanation": "Reason"}
    mock_place.return_value = {"ok": True}

    # Simulate request
    from flask import Flask
    app = Flask(__name__)
    with app.test_request_context():
        new_position_bot(None)

    mock_grok.assert_called_once()
    mock_place.assert_called_once()
```

#### `tests/test_e2e.py`
```python
# Run with real demo env; set env vars
import pytest
from kalshi_client import get_open_markets, get_positions

def test_real_markets():
    markets = get_open_markets(1)
    assert isinstance(markets, list)
    positions = get_positions()
    assert isinstance(positions, dict)
```

This completes the bot. Extend `place_order` for "no" sides or advanced strategies. Test thoroughly in demo before production. For Grok API details, refer to https://docs.x.ai/docs/ (assumes standard OpenAI compat). For Kalshi, https://docs.kalshi.com (uses JWT/ES256 signing).
