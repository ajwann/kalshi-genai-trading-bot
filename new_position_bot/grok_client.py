import requests
import json
import logging

logger = logging.getLogger(__name__)

class GrokClient:
    def __init__(self, api_key: str, model: str = "grok-4-1-fast-reasoning"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1/chat/completions"

    def analyze_market(self, market_data: dict) -> dict:
        """
        Sends market data to Grok and asks for a trading decision.
        Returns the JSON parsed response.
        """
        
        system_prompt = (
        "You are an expert prediction market trader, similar to 'Domer'. "
        "You analyze market metadata to find high-probability opportunities. "
        "You may recommend buying either the 'yes' side OR the 'no' side, "
        "or choose to pass if there isn't a good trade. "
        "Respond ONLY with a valid RFC 8259 JSON object containing exactly three keys: "
        "'ticker' (string or null), 'side' (\"yes\", \"no\", or null), and 'explanation' (string or null)."
        )


        user_content = (
        f"Analyze this prediction market and decide whether to take a position on the Yes side, "
        f"the No side, or skip the trade.\n\n"
        f"Market Title: {market_data.get('title')}\n"
        f"Ticker: {market_data.get('ticker')}\n"
        f"Subtitle: {market_data.get('subtitle')}\n"
        f"Category: {market_data.get('category')}\n"
        f"Current Yes Price: {market_data.get('yes_ask', 'N/A')}\n\n"
        "If you recommend a trade, return the ticker and which side to buy ('yes' or 'no'), "
        "with a short explanation. "
        "If you do NOT recommend a trade, return null for ticker and side, but still include an explanation.\n\n"
        "Example Yes Trade: {\"ticker\": \"KX-123\", \"side\": \"yes\", \"explanation\": \"Undervalued relative to polling data.\"}\n"
        "Example No Trade: {\"ticker\": \"KX-456\", \"side\": \"no\", \"explanation\": \"Market overestimates probability.\"}\n"
        "Example Pass: {\"ticker\": null, \"side\": null, \"explanation\": \"Not enough information.\"}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1, # Low temp for deterministic JSON
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            logger.info(f"Grok response content: {content}")
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
                
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Grok response: {content}")
            return {"ticker": None, "explanation": "JSON Error"}
        except Exception as e:
            logger.error(f"Grok API failed: {e}")
            return {"ticker": None, "explanation": str(e)}
