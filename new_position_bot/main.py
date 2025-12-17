import functions_framework
import logging
from utils import get_env_var, get_private_key
from kalshi_client import KalshiClient
from grok_client import GrokClient

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_bot_logic():
    logger.info("Starting New Position Bot run...")
    
    # 1. Configuration
    kalshi_base_url = get_env_var("KALSHI_BASE_URL", "https://demo-api.kalshi.co")
    kalshi_api_key = get_env_var("KALSHI_API_KEY", required=True)
    lookback_hours = int(get_env_var("LOOKBACK_HOURS", "1"))
    grok_model = get_env_var("GROK_MODEL", "grok-4-1-fast-reasoning")
    xai_api_key = get_env_var("XAI_API_KEY", required=True)
    
    try:
        kalshi_private_key = get_private_key()
    except Exception as e:
        logger.critical(str(e))
        return "Failed to load private key", 500

    # 2. Initialize Clients
    kalshi = KalshiClient(kalshi_base_url, kalshi_api_key, kalshi_private_key)
    grok = GrokClient(xai_api_key, grok_model)

    # 3. Fetch Data
    try:
        new_markets = kalshi.get_active_markets(created_after_hours=lookback_hours)
        viable_new_markets = [m for m in new_markets if (m['yes_ask'] < 100 and m['no_ask'] < 100)]
        current_positions = kalshi.get_positions()
        logger.info(f"Found {len(current_positions)} current positions.")
        
        # Extract tickers of current positions to filter
        held_tickers = {p['ticker'] for p in current_positions}
        
        logger.info(f"Found {len(viable_new_markets)} viable new active markets over the past {lookback_hours} hours.")
        
        for market in viable_new_markets:
            ticker = market['ticker']
            
            if ticker in held_tickers:
                logger.info(f"Skipping {ticker}, already held.")
                # TODO: block 1, uncomment after testing
                #continue
                # end block 1

            logger.info(f"Analyzing {ticker}: title: {market.get('title')}, yes_ask: {market.get('yes_ask')}, no_ask: {market.get('no_ask')}")
            
            # 4. Consult Grok
            recommendation = grok.analyze_market(market)
            
            rec_ticker = recommendation.get("ticker")
            explanation = recommendation.get("explanation")
            
            #TODO: uncomment below line after testing
            if 1 == 1: #rec_ticker and rec_ticker == ticker:
                logger.info(f"Grok recommends BUY on {ticker}. Reason: {explanation}")
                
                # 5. Execute Trade
                try:
                    # Defaulting to Buying 1 Yes Contract
                    logger.info(f"About to buy 1 yes contract for {ticker} at price {market.get('yes_ask')}")
                    order_response = kalshi.create_market_order(ticker, side="yes", count=1, price=market.get('yes_ask'))
                    logger.info(f"Order placed successfully: {order_response}")
                except Exception as trade_err:
                    logger.error(f"Failed to place order for {ticker}: {trade_err}")
            else:
                logger.info(f"Grok passed on {ticker}.")

    except Exception as e:
        logger.error(f"Bot execution failed: {e}")
        return f"Error: {e}", 500

    return "Run Complete", 200

# Cloud Function Entry Point
@functions_framework.http
def main(request):
    """HTTP Cloud Function."""
    result, status = run_bot_logic()
    return result, status

# Local Entry Point
if __name__ == "__main__":
    # For local testing, ensure env vars are set or load from .env
    from dotenv import load_dotenv
    load_dotenv()
    run_bot_logic()
