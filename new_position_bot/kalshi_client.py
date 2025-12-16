import time
import base64
import json
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class KalshiClient:
    def __init__(self, base_url: str, api_key: str, private_key_pem: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.private_key = load_pem_private_key(private_key_pem.encode(), password=None)

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        # Signature payload: timestamp + method + path (no query params for signature usually, but Kalshi specific)
        # Kalshi Docs: timestamp + method + path_without_query
        # Path should include the leading slash, e.g. /trade-api/v2/markets
        payload = f"{timestamp}{method}{path}".encode('utf-8')
        
        signature = self.private_key.sign(
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        path = f"/trade-api/v2{endpoint}"
        url = f"{self.base_url}{path}"
        timestamp = str(int(time.time() * 1000))
        
        signature = self._sign_request(method, path, timestamp)
        
        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }

        try:
            response = requests.request(method, url, headers=headers, params=params, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"API Error: {e.response.text}")
            raise

    def get_active_markets(self, created_after_hours: int = 1) -> List[Dict]:
        """
        Fetches markets created in the last N hours that are currently active.
        """
        # Kalshi doesn't strictly support filtering by 'created_time' in the GET params 
        # for all endpoints, so we fetch open markets and filter client-side.
        # We use limit=500 to get a good chunk of recent activity.
        params = {"status": "open", "limit": 500}
        data = self._request("GET", "/markets", params=params)
        
        markets = data.get("markets", [])
        filtered_markets = []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=created_after_hours)
        
        for m in markets:
            # created_time format example: "2023-11-07T05:31:56Z"
            created_str = m.get("created_time")
            if created_str:
                # Handle potential trailing Z or fractional seconds
                try:
                    created_dt = datetime.strptime(created_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    if created_dt >= cutoff_time:
                        filtered_markets.append(m)
                except ValueError:
                    # Attempt ISO format with microseconds if basic parse fails
                    try:
                         created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                         if created_dt.replace(tzinfo=None) >= cutoff_time:
                             filtered_markets.append(m)
                    except Exception as e:
                        logger.warning(f"Failed to parse date {created_str}: {e}")
                        continue
                        
        return filtered_markets

    def get_positions(self) -> List[Dict]:
        """Returns a list of positions held by the user."""
        data = self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    def create_market_order(self, ticker: str, side: str = "yes", count: int = 1, price: int = 99):
        """Places a market order."""
        payload = {
            "ticker": ticker,
            "action": "buy",
            "type": "market",
            "side": side,
            "count": count,
            "yes_price": price, # TODO: need to get the bid or ask price from the /market endpoint, also need to send the /market endpoint response to Grok
            "cancel_order_on_pause": true,
            "client_order_id": str(int(time.time() * 1000000)) # Unique ID
        }
        logger.info(f"Placing order: {payload}")
        return self._request("POST", "/portfolio/orders", data=payload)
