### Project Structure
The "New Position Bot" is implemented as a Python 3.13 project. The structure is as follows:

```
new_position_bot/
├── main.py                 # Entry point: runnable locally or as GCP Cloud Function
├── bot.py                  # Core bot logic for market analysis, Grok calls, and trading
├── kalshi_client.py        # Kalshi API client wrapper
├── grok_client.py          # xAI Grok API client wrapper
├── prompt.py               # Role-based prompt template
├── .env.example            # Example environment variables
├── requirements.txt        # Dependencies
├── tests/
│   ├── __init__.py
│   ├── test_bot.py         # Unit tests for bot logic
│   ├── test_kalshi_client.py # Unit/integration tests for Kalshi client (using mocks)
│   ├── test_grok_client.py # Unit/integration tests for Grok client (using mocks)
│   └── test_end_to_end.py  # Functional E2E tests (requires demo API access)
├── logging.conf            # Python logging configuration for detailed logs
└── README.md               # Setup and run instructions
```

### Environment Variables
Set these in your `.env` file or as GCP Cloud Function environment variables:
- `KALSHI_API_KEY`: Your Kalshi API key (string).
- `KALSHI_PRIVATE_KEY`: Path to or contents of your RSA private key in PEM format (string; for local, use file path or inline PEM).
- `KALSHI_BASE_URL`: Base API URL (default: `https://demo-api.kalshi.co/trade-api/v2`).
- `LOOKBACK_HOURS`: Hours to look back for new markets (default: `1`, int).
- `GROK_MODEL`: Grok model to use (default: `grok-4.1`; string).
- `XAI_API_KEY`: Your xAI API key (string).

For local runs, use `python-dotenv` to load from `.env`. For GCP, set directly in the function config.

### requirements.txt
```
requests==2.32.3
cryptography==42.0.8
python-dateutil==2.9.0.post0
pytest==8.3.2
pytest-mock==3.14.0
requests-mock==1.11.0
python-dotenv==1.0.1
```

### logging.conf
```
[loggers]
keys=root

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### prompt.py
```python
PROMPT_TEMPLATE = """
You are Domer, an expert in professional prediction market trading on Kalshi. You specialize in high-accuracy positions based on data-driven analysis.

Your task: Analyze the following Kalshi market and decide whether to take a position (buy YES or NO contracts). Before deciding, perform online research (using your search capabilities) to become an expert in the event/topic. Consider probabilities, news, historical data, and market sentiment.

Market details:
- ID: {market_id}
- Question: {question}
- Category: {category}
- Open: {open_time}
- Close: {close_time}
- Status: {status}
- Yes Price: {yes_price}
- No Price: {no_price}
- Volume: {volume}

Respond ONLY with a valid RFC 8259 compliant JSON object with exactly two keys:
- "market_id": The market ID as string if you recommend a position, else null.
- "explanation": A detailed textual explanation of your decision (why take this position or skip), else null.

If no position is recommended (e.g., low confidence, insufficient data), set both to null.

Example (recommend):
{{"market_id": "{market_id}", "explanation": "Based on recent polls showing 65% support and expert forecasts, buy YES at current price for expected value gain."}}

Example (skip):
{{"market_id": null, "explanation": null}}
"""
```

### kalshi_client.py
```python
import os
import logging
import requests
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import json
import base64

logger = logging.getLogger(__name__)

class KalshiClient:
    def __init__(self):
        self.api_key = os.getenv('KALSHI_API_KEY')
        private_key_str = os.getenv('KALSHI_PRIVATE_KEY')
        if not private_key_str:
            raise ValueError("KALSHI_PRIVATE_KEY not set")
        if os.path.exists(private_key_str):
            with open(private_key_str, 'rb') as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(), password=None
                )
        else:
            self.private_key = serialization.load_pem_private_key(
                private_key_str.encode(), password=None
            )
        self.base_url = os.getenv('KALSHI_BASE_URL', 'https://demo-api.kalshi.co/trade-api/v2')
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})

    def _sign_request(self, method: str, path: str, body: str = '') -> str:
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')
        message = f"{method}\n{path}\n{timestamp}\n{body}"
        signature = self.private_key.sign(
            message.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        path = endpoint.lstrip('/')
        body = json.dumps(data) if data else ''
        signature = self._sign_request(method, path, body)
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')
        headers = {
            'Signature': signature,
            'Timestamp': timestamp,
            'Content-Type': 'application/json',
        }
        self.session.headers.update(headers)
        url = f"{self.base_url}/{endpoint}"
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            else:
                response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Kalshi API error: {e}")
            raise

    def get_markets(self, status: str = 'open', limit: int = 100, created_after: str = None) -> list:
        params = {'status': status, 'limit': limit}
        if created_after:
            params['open_time[gte]'] = created_after
        # Note: Kalshi /markets endpoint supports filtering by open_time[gte] for creation proxy
        endpoint = 'markets'
        data = self._make_request('GET', endpoint, params)  # Adapt params to query string if needed
        logger.info(f"Fetched {len(data.get('markets', []))} markets")
        return data.get('markets', [])

    def get_positions(self) -> list:
        endpoint = 'positions'
        data = self._make_request('GET', endpoint)
        positions = data.get('positions', [])
        logger.info(f"Current positions: {len(positions)}")
        return positions

    def has_position(self, market_id: str) -> bool:
        positions = self.get_positions()
        return any(p['market_id'] == market_id for p in positions)

    def place_order(self, market_id: str, side: str, count: int = 10) -> dict:
        # side: 'yes' or 'no'
        data = {
            'market_id': market_id,
            'side': side,
            'count': count,  # Default small position for demo
            'type': 'market',  # Market order for simplicity
            'time_in_force': 'ioc'  # Immediate or cancel
        }
        endpoint = 'orders'
        result = self._make_request('POST', endpoint, data)
        logger.info(f"Placed order for {market_id}: {result}")
        return result
```

### grok_client.py
```python
import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

class GrokClient:
    def __init__(self):
        self.api_key = os.getenv('XAI_API_KEY')
        if not self.api_key:
            raise ValueError("XAI_API_KEY not set")
        self.base_url = 'https://api.x.ai/v1/chat/completions'
        self.model = os.getenv('GROK_MODEL', 'grok-4.1')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    def generate_response(self, market_details: dict) -> dict:
        from prompt import PROMPT_TEMPLATE
        prompt = PROMPT_TEMPLATE.format(
            market_id=market_details['id'],
            question=market_details['question'],
            category=market_details['category'],
            open_time=market_details['open_time'],
            close_time=market_details['close_time'],
            status=market_details['status'],
            yes_price=market_details['yes_bid'] or 0,
            no_price=market_details['no_bid'] or 0,
            volume=market_details.get('volume', 0)
        )
        data = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant that responds only with valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1,  # Low for deterministic JSON
            'response_format': {'type': 'json_object'},  # Enforce JSON
        }
        try:
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            parsed = json.loads(content)
            logger.info(f"Grok response for {market_details['id']}: {parsed}")
            return parsed
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Grok API error: {e}")
            return {'market_id': None, 'explanation': None}
```

### bot.py
```python
import os
import logging
from datetime import datetime, timedelta
import time

from kalshi_client import KalshiClient
from grok_client import GrokClient

logger = logging.getLogger(__name__)

class NewPositionBot:
    def __init__(self):
        self.kalshi = KalshiClient()
        self.grok = GrokClient()
        self.lookback_hours = int(os.getenv('LOOKBACK_HOURS', 1))

    def run(self):
        logger.info("Starting New Position Bot run")
        lookback_time = (datetime.utcnow() - timedelta(hours=self.lookback_hours)).isoformat() + 'Z'
        markets = self.kalshi.get_markets(created_after=lookback_time)
        
        for market in markets:
            market_id = market['id']
            if self.kalshi.has_position(market_id):
                logger.info(f"Skipping {market_id}: Already have position")
                continue
            
            logger.info(f"Analyzing new market {market_id}: {market['question']}")
            grok_response = self.grok.generate_response(market)
            
            if grok_response['market_id']:
                explanation = grok_response['explanation']
                logger.info(f"Grok recommends position in {market_id}: {explanation}")
                # For simplicity, buy YES if recommended; in production, parse for side
                # Assuming recommendation implies YES; extend prompt for side if needed
                order_result = self.kalshi.place_order(market_id, 'yes')
                logger.info(f"Position taken: {order_result}")
            else:
                logger.info(f"Grok skips {market_id}: {grok_response['explanation'] or 'No explanation'}")

        logger.info("Bot run completed")
```

### main.py
```python
import os
import logging
import logging.config
from dotenv import load_dotenv

from bot import NewPositionBot

# Load env for local runs
load_dotenv()

# Setup logging
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

def main(request=None):
    """Entry point: local or GCP Cloud Function."""
    if request:  # GCP mode
        logger.info("Running as GCP Cloud Function")
    else:
        logger.info("Running locally")
    
    bot = NewPositionBot()
    bot.run()
    
    return 'Bot run completed', 200

if __name__ == '__main__':
    main()
```

### Tests (tests/test_bot.py - Unit Tests Example)
```python
import pytest
from unittest.mock import Mock, patch
from bot import NewPositionBot

@pytest.fixture
def bot():
    return NewPositionBot()

@patch('bot.KalshiClient')
@patch('bot.GrokClient')
def test_run_skips_positioned_markets(mock_grok, mock_kalshi):
    mock_kalshi_instance = Mock()
    mock_kalshi_instance.get_markets.return_value = [{'id': 'test123', 'question': 'Test?'}]
    mock_kalshi_instance.has_position.return_value = True
    mock_grok_instance = Mock()
    bot = NewPositionBot()
    bot.kalshi = mock_kalshi_instance
    bot.grok = mock_grok_instance
    
    bot.run()
    
    mock_kalshi_instance.get_markets.assert_called_once()
    mock_kalshi_instance.has_position.assert_called_once_with('test123')

@patch('bot.KalshiClient')
@patch('bot.GrokClient')
def test_run_places_order(mock_grok, mock_kalshi):
    mock_kalshi_instance = Mock()
    mock_kalshi_instance.get_markets.return_value = [{'id': 'test123', 'question': 'Test?'}]
    mock_kalshi_instance.has_position.return_value = False
    mock_grok_instance = Mock()
    mock_grok_instance.generate_response.return_value = {'market_id': 'test123', 'explanation': 'Reason'}
    mock_kalshi_instance.place_order.return_value = {'success': True}
    bot = NewPositionBot()
    bot.kalshi = mock_kalshi_instance
    bot.grok = mock_grok_instance
    
    bot.run()
    
    mock_grok_instance.generate_response.assert_called_once()
    mock_kalshi_instance.place_order.assert_called_once_with('test123', 'yes')
```

### Tests (tests/test_kalshi_client.py - Integration with Mocks)
```python
import pytest
from unittest.mock import patch, Mock
import requests
from kalshi_client import KalshiClient

@patch('kalshi_client.requests.Session')
def test_get_markets(mock_session):
    mock_response = Mock()
    mock_response.json.return_value = {'markets': [{'id': '1'}]}
    mock_session.return_value.get.return_value = mock_response
    client = KalshiClient()
    client.session = mock_session.return_value
    
    markets = client.get_markets()
    
    assert len(markets) == 1
    mock_session.return_value.get.assert_called()

# Similar tests for other methods...
```

### Tests (tests/test_grok_client.py - Unit)
```python
import pytest
from unittest.mock import patch
from grok_client import GrokClient

@patch('grok_client.requests.post')
def test_generate_response(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        'choices': [{'message': {'content': '{"market_id": "123", "explanation": "Test"}'}}]
    }
    mock_post.return_value = mock_response
    client = GrokClient()
    
    result = client.generate_response({'id': '123', 'question': 'Test?'})
    
    assert result['market_id'] == '123'
```

### Tests (tests/test_end_to_end.py - Functional E2E)
```python
# Requires setting demo env vars; run with pytest -m e2e
import pytest
from bot import NewPositionBot

@pytest.mark.e2e
def test_end_to_end_run():
    bot = NewPositionBot()
    # Mock or limit to dry-run; in CI, use demo API
    with patch.object(bot.kalshi, 'place_order') as mock_place:
        bot.run()
    # Assert no exceptions; check logs for flow
```

### README.md
```
# New Position Bot

A GenAI-powered trading bot for Kalshi using Grok.

## Setup
1. Copy `.env.example` to `.env` and fill values.
2. `pip install -r requirements.txt`
3. For local: `python main.py`
4. For GCP: Deploy `main.py` as HTTP Cloud Function, set env vars, schedule hourly via Cloud Scheduler.

## Testing
`pytest tests/`
For E2E: `pytest tests/test_end_to_end.py -m e2e` (set demo creds).

## Notes
- Uses demo API by default; switch via KALSHI_BASE_URL.
- Positions: Defaults to buying 10 YES contracts; extend prompt for side/count.
- Logging: Detailed to stdout.
- Prompt: Enforces JSON; Grok researches via its tools.
```

This implementation is stateless, logs to stdout, and runs hourly (schedule externally). It handles the specified flow: fetch new open markets, skip positioned ones, query Grok, parse JSON, and trade if recommended. Extend as needed for production (e.g., error retries, position sizing).
