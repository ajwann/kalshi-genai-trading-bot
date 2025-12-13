### Project Structure
The "New Position Bot" is structured as a Python project suitable for local execution or deployment as a Google Cloud Function. Here's the recommended directory layout:

```
new_position_bot/
├── main.py                 # Main entry point (runnable locally or as Cloud Function)
├── bot.py                  # Core bot logic (stateless)
├── kalshi_client.py        # Kalshi API client wrapper
├── grok_client.py          # Grok API client wrapper
├── prompts.py              # GenAI role-based prompt template
├── requirements.txt        # Dependencies
├── tests/                  # Pytest tests
│   ├── __init__.py
│   ├── test_bot.py         # Unit tests for bot logic
│   ├── test_kalshi.py      # Integration tests for Kalshi client (mocked)
│   └── test_end_to_end.py  # Functional E2E tests (mocked APIs)
├── kalshi_private_key.pem  # Local RSA private key file (gitignore this in production)
├── .env.example            # Example env file
└── .gitignore              # Standard ignores (e.g., .env, __pycache__)
```

### Environment Variables
Create a `.env` file locally (based on `.env.example`):
```
KALSHI_API_KEY=your_kalshi_api_key
KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
LOOKBACK_HOURS=1
GROK_MODEL=grok-4.1  # Or other models like grok-3
XAI_API_KEY=your_xai_api_key
GCP_PROJECT_ID=your_gcp_project  # For Secret Manager (optional, detected in Cloud Function)
```

Load via `python-dotenv` for local runs.

### requirements.txt
```
requests==2.32.3
cryptography==42.0.8  # For RSA signing
python-dotenv==1.0.0  # Local env loading
pytest==8.3.2
pytest-mock==3.14.0
google-cloud-secret-manager==2.20.0  # For GCP Secret Manager
```

### kalshi_client.py
This stateless client handles authentication (RSA signing) and key endpoints: listing markets, getting positions, placing orders. Uses `requests`. Authentication follows Kalshi's header-based signing (time, path, body).

```python
import os
import time
import hashlib
import requests
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class KalshiClient:
    def __init__(self, api_key: str, base_url: str, private_key_pem: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})

    def _sign_request(self, method: str, path: str, body: str = '') -> str:
        timestamp = str(int(time.time()))
        prehash = f"{method}{path}{body}{timestamp}"
        digest = hashlib.sha256(prehash.encode()).digest()
        signature = self.private_key.sign(
            digest,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return f"{timestamp}:{signature.hex()}"

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        path = endpoint.lstrip('/')
        body = kwargs.get('json', {}) if method == 'POST' else ''
        if isinstance(body, dict):
            body = ''  # For GET/LIST, no body
        sig = self._sign_request(method.upper(), f'/{path}', str(body))
        self.session.headers['kalshi-signature'] = sig
        url = f"{self.base_url}/{path}"
        try:
            resp = self.session.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Kalshi API error: {e}")
            return None

    def list_markets(self, status: str = 'open', limit: int = 100, created_after: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {'status': status, 'limit': limit}
        if created_after:
            params['created_after'] = created_after  # ISO format
        data = self._make_request('GET', 'markets', params=params)
        return data.get('markets', []) if data else []

    def get_positions(self) -> List[Dict[str, Any]]:
        data = self._make_request('GET', 'positions')
        return data.get('positions', []) if data else []

    def place_order(self, market_id: str, side: str, count: int, price: float, type: str = 'market') -> Optional[Dict[str, Any]]:
        payload = {
            'market_id': market_id,
            'side': side,  # 'yes' or 'no'
            'count': count,
            'price': price,
            'type': type
        }
        return self._make_request('POST', 'orders', json=payload)

def load_private_key() -> str:
    if os.getenv('FUNCTION_NAME'):  # GCP Cloud Function detection
        from google.cloud import secretmanager
        project_id = os.getenv('GCP_PROJECT_ID', os.environ.get('GOOGLE_CLOUD_PROJECT'))
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/kalshi-rsa-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode('UTF-8')
    else:
        with open('kalshi_private_key.pem', 'r') as f:
            return f.read()
```

### grok_client.py
Wrapper for xAI Grok API. Assumes OpenAI-compatible endpoint `/v1/chat/completions` (based on standard practices; adjust if docs differ). Forces JSON mode for RFC 8259 compliance.

```python
import os
import requests
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GrokClient:
    def __init__(self, api_key: str, model: str = 'grok-4.1'):
        self.api_key = api_key
        self.model = model
        self.base_url = 'https://api.x.ai/v1'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def generate_response(self, prompt: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            'response_format': {'type': 'json_object'}  # Forces JSON schema
        }
        try:
            resp = requests.post(f'{self.base_url}/chat/completions', json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content']
            # Parse JSON from content
            import json
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Grok API error: {e}")
            return None
```

### prompts.py
Role-based system prompt for Grok.

```python
SYSTEM_PROMPT = """
You are Domer, an expert in professional prediction market trading on platforms like Kalshi. You specialize in high-accuracy, data-driven positions in binary event markets.

When given a market description, perform online research (using your tools) to become an expert: search recent news, polls, expert analyses, historical data, and real-time events relevant to the market question. Analyze probabilities rigorously.

Respond ONLY with a valid RFC 8259 JSON object with exactly two keys:
- "market_id": the market's unique ID (string) if you recommend entering a position, else null.
- "explanation": a concise textual explanation of your decision (why enter or skip), else null.

If no position is warranted (e.g., insufficient data, low edge), return {"market_id": null, "explanation": null}.

Example for entry: {"market_id": "123", "explanation": "Based on recent polls showing 65% support and historical trends, buy YES at current prices for 10% edge."}

Example for skip: {"market_id": null, "explanation": null}
"""
```

### bot.py
Core stateless logic. Analyzes markets, queries Grok, places orders if recommended. Logs to stdout.

```python
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from kalshi_client import KalshiClient, load_private_key
from grok_client import GrokClient
from prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_bot() -> None:
    api_key = os.getenv('KALSHI_API_KEY')
    base_url = os.getenv('KALSHI_BASE_URL', 'https://demo-api.kalshi.co/trade-api/v2')
    lookback_hours = int(os.getenv('LOOKBACK_HOURS', '1'))
    grok_model = os.getenv('GROK_MODEL', 'grok-4.1')
    xai_api_key = os.getenv('XAI_API_KEY')

    if not all([api_key, xai_api_key]):
        logger.error("Missing required env vars")
        return

    private_key_pem = load_private_key()
    kalshi = KalshiClient(api_key, base_url, private_key_pem)
    grok = GrokClient(xai_api_key, grok_model)

    # Get lookback timestamp
    lookback_time = datetime.utcnow() - timedelta(hours=lookback_hours)
    created_after = lookback_time.isoformat() + 'Z'  # Kalshi expects ISO with Z

    # Fetch new open markets
    markets = kalshi.list_markets(status='open', created_after=created_after)
    logger.info(f"Found {len(markets)} new open markets")

    # Get current positions
    positions = kalshi.get_positions()
    held_markets = {pos['market_id'] for pos in positions}
    logger.info(f"Current positions in {len(held_markets)} markets")

    for market in markets:
        market_id = market['id']  # Assuming 'id' key from docs
        if market_id in held_markets:
            logger.info(f"Skipping held market {market_id}")
            continue

        # Query Grok
        user_prompt = f"Analyze this Kalshi market: {market['question']} (ID: {market_id}). Current yes_price: {market.get('yes_bid', 'N/A')}, volume: {market.get('volume', 'N/A')}. Recommend position or skip."
        grok_resp = grok.generate_response(user_prompt, SYSTEM_PROMPT)
        if not grok_resp:
            logger.warning(f"Grok failed for {market_id}")
            continue

        market_id_rec = grok_resp.get('market_id')
        explanation = grok_resp.get('explanation')
        if market_id_rec is None or explanation is None:
            logger.info(f"Grok skips {market_id}: {explanation or 'No reason'}")
            continue

        if market_id_rec != market_id:
            logger.warning(f"Grok mismatch: expected {market_id}, got {market_id_rec}")
            continue

        # Place order (example: buy 10 YES at market price if recommended; adjust based on explanation)
        # Parse explanation for side/count/price (simplified: assume YES, count=10, price=market yes_ask)
        side = 'yes'  # Default; parse from explanation if needed
        count = 10
        price = market.get('yes_ask', 0.5)  # Use current ask
        order = kalshi.place_order(market_id, side, count, price)
        if order:
            logger.info(f"Placed order for {market_id}: {order}. Reason: {explanation}")
        else:
            logger.error(f"Failed to place order for {market_id}")
```

### main.py
Entry point. Runs locally or as Cloud Function.

```python
import os
import sys
from bot import run_bot
from dotenv import load_dotenv

if __name__ == '__main__':
    if not os.getenv('FUNCTION_NAME'):  # Local run
        load_dotenv()
    run_bot()

# For GCP Cloud Function (entrypoint in gcf.yaml or inline)
def gcf_entry(request):
    run_bot()
    return 'Bot executed successfully', 200
```

### Tests (in tests/)
#### test_bot.py (Unit tests)
```python
import pytest
from unittest.mock import Mock, patch
from bot import run_bot

@patch('bot.KalshiClient')
@patch('bot.GrokClient')
def test_run_bot_skip_held(mocked_grok, mocked_kalshi):
    mocked_kalshi.return_value.list_markets.return_value = [{'id': '123', 'question': 'Test?'}]
    mocked_kalshi.return_value.get_positions.return_value = [{'market_id': '123'}]
    run_bot()
    mocked_grok.assert_not_called()  # Skips held
```

#### test_kalshi.py (Integration, mocked HTTP)
```python
import pytest
from unittest.mock import patch
from kalshi_client import KalshiClient

@patch('requests.Session.request')
def test_list_markets(mocked_request):
    mocked_request.return_value.json.return_value = {'markets': [{'id': '123'}]}
    client = KalshiClient('key', 'url', 'pem')
    markets = client.list_markets(created_after='2023-01-01T00:00:00Z')
    assert len(markets) == 1
```

#### test_end_to_end.py (Functional, full mocks)
```python
import pytest
from unittest.mock import patch, MagicMock
from bot import run_bot

@patch('bot.load_private_key')
@patch('bot.KalshiClient.place_order')
@patch('bot.GrokClient.generate_response')
@patch('bot.KalshiClient.get_positions')
@patch('bot.KalshiClient.list_markets')
def test_e2e_recommend_buy(mocked_list, mocked_pos, mocked_grok, mocked_place, mocked_key):
    mocked_list.return_value = [{'id': '123', 'question': 'Test?'}]
    mocked_pos.return_value = []
    mocked_grok.return_value = {'market_id': '123', 'explanation': 'Buy YES'}
    mocked_place.return_value = {'status': 'filled'}
    with patch.dict('os.environ', {'KALSHI_API_KEY': 'key', 'XAI_API_KEY': 'key'}):
        run_bot()
    mocked_place.assert_called_once()
```

Run tests: `pytest tests/ -v`

### Deployment Notes
- **Local**: `python main.py`
- **GCP Cloud Function**: Deploy via `gcloud functions deploy new-position-bot --runtime python313 --trigger-http --entry-point gcf_entry --timeout 540s --source .` (schedule via Cloud Scheduler every hour).
- **Logging**: Outputs to stdout; GCP captures to Cloud Logging.
- **Config**: Set env vars in GCP console. Store private key in Secret Manager as `kalshi-rsa-key`.
- **Assumptions**: Market keys (`id`, `question`, `yes_ask`) based on typical Kalshi docs. Adjust signing if exact Kalshi spec differs (e.g., body hashing). Grok endpoint assumed OpenAI-compatible; update from https://docs.x.ai if needed. For production, add retries, validation, and risk limits.
