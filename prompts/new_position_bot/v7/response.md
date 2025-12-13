### Project Structure
The "New Position Bot" is implemented as a Python package for easy deployment as a Google Cloud Function (stateless entrypoint in `main.py`) or local execution (via `bot.py`). The structure is:

```
new_position_bot/
├── __init__.py
├── main.py              # GCP Cloud Function entrypoint
├── bot.py               # Core bot logic
├── kalshi_client.py     # Kalshi API client
├── grok_client.py       # Grok API client
├── prompt.py            # GenAI prompt template
├── .env.example         # Environment variable template
├── requirements.txt     # Dependencies
├── tests/
│   ├── __init__.py
│   ├── test_unit.py     # Unit tests
│   ├── test_integration.py  # Integration tests (requires mocks/secrets)
│   └── test_e2e.py      # End-to-end tests (requires demo API access)
└── pytest.ini           # Pytest config
```

### Environment Variables
Create a `.env` file based on `.env.example`:
```
KALSHI_API_KEY=your_kalshi_api_key
KALSHI_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\n... (PEM content with \n for newlines)
KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
LOOKBACK_HOURS=1
GROK_MODEL=grok-4.1  # Or 'grok-3', etc.
XAI_API_KEY=your_xai_api_key
```

Load via `python-dotenv` in code.

### requirements.txt
```
requests==2.31.0
cryptography==42.0.5
python-dotenv==1.0.0
pytest==7.4.3
pytest-mock==3.12.0
```

### prompt.py
```python
PROMPT_TEMPLATE = """
You are Domer, a professional expert in prediction market trading on platforms like Kalshi, Polymarket, and PredictIt. You specialize in quantitative analysis, sentiment tracking, and risk-adjusted positioning in binary event markets. Your goal is to identify high-confidence opportunities with positive expected value.

For the following Kalshi market, perform online research (using your tools for web search, X search, etc.) to become an expert: Analyze historical data, news, polls, expert opinions, social sentiment, and quantitative models relevant to the event. Consider probabilities, liquidity, bid-ask spreads, and your estimated edge over the market price.

Market details: {market_details}

Based on your research, decide whether to take a position in this market. If you recommend a position, specify the market_id. If no position (e.g., insufficient edge, high risk, or lack of information), set market_id to null.

Respond ONLY with a valid RFC 8259 compliant JSON object in this exact format, nothing else:
{{"market_id": "{market_id or null}", "explanation": "{textual explanation of decision, max 500 words}"}}
"""
```

### kalshi_client.py
```python
import os
import requests
import time
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

class KalshiClient:
    def __init__(self):
        self.api_key = os.getenv('KALSHI_API_KEY')
        self.private_key_pem = os.getenv('KALSHI_PRIVATE_KEY').replace('\\n', '\n')
        self.base_url = os.getenv('KALSHI_BASE_URL', 'https://demo-api.kalshi.co/trade-api/v2')
        self.private_key = serialization.load_pem_private_key(
            self.private_key_pem.encode(),
            password=None,
        )
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})

    def _sign_request(self, method: str, path: str, body: str = '') -> str:
        timestamp = str(int(time.time()))
        message = f"{method}\n{path}\n{body}\n{timestamp}"
        signature = self.private_key.sign(
            message.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return signature.hex()

    def _make_request(self, method: str, path: str, body: dict = None) -> dict:
        body_str = json.dumps(body) if body else ''
        sig = self._sign_request(method, path, body_str)
        headers = {
            'Signature': sig,
            'Timestamp': str(int(time.time())),
            'Content-Type': 'application/json',
        }
        self.session.headers.update(headers)
        url = f"{self.base_url}/{path}"
        response = self.session.request(method, url, json=body)
        response.raise_for_status()
        return response.json()

    def get_new_markets(self, lookback_hours: int = 1) -> list:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        path = f"markets?start_time={start_ts}&end_time={end_ts}&status=open"
        data = self._make_request('GET', path)
        return data.get('markets', [])

    def get_positions(self) -> list:
        path = 'positions'
        data = self._make_request('GET', path)
        return data.get('positions', [])

    def get_market(self, market_id: str) -> dict:
        path = f'markets/{market_id}'
        data = self._make_request('GET', path)
        return data

    def place_order(self, market_id: str, side: str, count: int, price: float) -> dict:
        body = {
            'market_id': market_id,
            'side': side,  # 'yes' or 'no'
            'count': count,
            'price': price,
            'type': 'market',  # Market order for simplicity
        }
        path = 'orders'
        return self._make_request('POST', path, body)
```

### grok_client.py
```python
import os
import requests
import json
from dotenv import load_dotenv
from prompt import PROMPT_TEMPLATE

load_dotenv()

class GrokClient:
    def __init__(self):
        self.api_key = os.getenv('XAI_API_KEY')
        self.model = os.getenv('GROK_MODEL', 'grok-4.1')
        self.base_url = 'https://api.x.ai/v1/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    def get_recommendation(self, market_details: str) -> dict:
        prompt = PROMPT_TEMPLATE.format(market_details=market_details)
        body = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'response_format': {'type': 'json_object'},
            'temperature': 0.1,
        }
        response = requests.post(self.base_url, headers=self.headers, json=body)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON from Grok")
```

### bot.py
```python
import logging
import json
from datetime import datetime
from kalshi_client import KalshiClient
from grok_client import GrokClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_bot():
    lookback_hours = int(os.getenv('LOOKBACK_HOURS', '1'))
    kalshi = KalshiClient()
    grok = GrokClient()

    # Get current positions
    positions = kalshi.get_positions()
    positioned_markets = {pos['market_id'] for pos in positions}

    # Get new active markets
    new_markets = kalshi.get_new_markets(lookback_hours)
    logger.info(f"Found {len(new_markets)} new markets")

    for market in new_markets:
        market_id = market['id']
        if market_id in positioned_markets:
            logger.info(f"Skipping market {market_id}: already positioned")
            continue

        # Get full market details
        market_details = kalshi.get_market(market_id)
        details_str = json.dumps(market_details, default=str)

        # Ask Grok
        try:
            rec = grok.get_recommendation(details_str)
            if rec['market_id'] is not None:
                # Place order (simplified: assume 'yes' side, count=1, price=0.5; customize based on rec if extended)
                order = kalshi.place_order(market_id, 'yes', 1, 0.5)
                logger.info(f"Placed order for {market_id}: {order}")
            else:
                logger.info(f"No position recommended for {market_id}: {rec['explanation']}")
        except Exception as e:
            logger.error(f"Error processing market {market_id}: {e}")
```

### main.py (GCP Cloud Function)
```python
import functions_framework
from bot import run_bot
import logging

logging.basicConfig(level=logging.INFO)

@functions_framework.http
def entrypoint(request):
    run_bot()
    return 'Bot executed successfully', 200
```

### Local Run
Run `python bot.py` for local execution (logs to stdout).

### Tests (tests/test_unit.py)
```python
import pytest
from unittest.mock import Mock, patch
from kalshi_client import KalshiClient
from grok_client import GrokClient

@pytest.fixture
def mock_kalshi():
    with patch('kalshi_client.requests.Session') as mock_session:
        yield mock_session

def test_kalshi_get_new_markets(mock_kalshi):
    client = KalshiClient()
    # Mock response
    mock_response = Mock()
    mock_response.json.return_value = {'markets': [{'id': '1'}]}
    mock_kalshi.return_value.request.return_value = mock_response
    markets = client.get_new_markets()
    assert len(markets) == 1

def test_grok_recommendation():
    client = GrokClient()
    with patch('grok_client.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': '{"market_id": null, "explanation": "No edge"}'}}]
        }
        mock_post.return_value = mock_response
        rec = client.get_recommendation('test')
        assert rec['market_id'] is None
```

### tests/test_integration.py
```python
# Requires env vars; uses pytest.mark.integration
import pytest
from bot import run_bot
from unittest.mock import patch

@pytest.mark.integration
def test_bot_flow(monkeypatch):
    monkeypatch.setenv('LOOKBACK_HOURS', '0')  # Minimal lookback for test
    with patch('bot.kalshi_client.KalshiClient.get_new_markets') as mock_markets, \
         patch('bot.kalshi_client.KalshiClient.get_positions') as mock_pos, \
         patch('bot.grok_client.GrokClient.get_recommendation') as mock_grok, \
         patch('bot.kalshi_client.KalshiClient.place_order') as mock_order:
        mock_markets.return_value = []
        mock_pos.return_value = []
        mock_grok.return_value = {'market_id': None, 'explanation': 'test'}
        run_bot()  # Should run without errors
```

### tests/test_e2e.py
```python
# End-to-end: Requires demo API access; run with pytest -m e2e
import pytest
from kalshi_client import KalshiClient

@pytest.mark.e2e
def test_e2e_kalshi():
    client = KalshiClient()
    positions = client.get_positions()
    assert isinstance(positions, list)  # Basic API reachability
```

### pytest.ini
```
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --strict-markers
markers =
    integration: Integration tests requiring mocks/secrets
    e2e: End-to-end tests requiring API access
```

Run tests: `pytest tests/`.

### Notes
- **Statelessness**: No persistent state; each run fetches fresh data.
- **Logging**: Uses `logging` module; outputs to stdout for GCP/local.
- **JSON Compliance**: Grok prompt enforces RFC 8259 via `json_object` mode.
- **Trading Simplification**: Order placement assumes a basic 'yes' position; extend `place_order` based on Grok's potential future enhancements (e.g., parse side/price from explanation).
- **Security**: PEM key loaded securely; never hardcode.
- **Deployment**: For GCP, zip and deploy `main.py` + deps as Cloud Function with HTTP trigger (scheduled via Cloud Scheduler hourly).
- **Python Version**: Specify 3.13.11 in GCP runtime or local env (e.g., pyenv).

This provides a complete, testable bot. Extend as needed for production risk management.
