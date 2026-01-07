import functions_framework
import logging
from utils import get_env_var, get_private_key
from kalshi_client import KalshiClient
from grok_client import GrokClient
from spending_limit import SpendingLimitTracker

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
    
    # Spending limit configuration
    spending_limit_period_hours = int(get_env_var("SPENDING_LIMIT_PERIOD_HOURS", "24"))
    spending_limit_cents = int(get_env_var("SPENDING_LIMIT_CENTS", "10000"))  # Default $100.00
    bot_identifier = get_env_var("BOT_IDENTIFIER", "NEW_POSITION_BOT")
    
    try:
        kalshi_private_key = get_private_key()
    except Exception as e:
        logger.critical(str(e))
        return "Failed to load private key", 500

    # 2. Initialize Clients
    kalshi = KalshiClient(kalshi_base_url, kalshi_api_key, kalshi_private_key)
    grok = GrokClient(xai_api_key, grok_model)
    spending_tracker = SpendingLimitTracker(
        limit_period_hours=spending_limit_period_hours,
        max_spending_cents=spending_limit_cents
    )

    # 3. Fetch Data
    try:
        # Get recent orders to calculate current spending
        try:
            all_orders = kalshi.get_orders(bot_identifier=bot_identifier)
            current_spending = spending_tracker.calculate_current_spending(all_orders)
            logger.info(f"Current bot spending in last {spending_limit_period_hours} hours: ${current_spending/100:.2f} (limit: ${spending_limit_cents/100:.2f})")
        except Exception as e:
            logger.warning(f"Failed to fetch orders for spending calculation: {e}")
            current_spending = 0  # Assume no spending if we can't fetch
        
        new_markets = kalshi.get_active_markets(created_after_hours=lookback_hours)
        viable_new_markets = [m for m in new_markets if (m['yes_ask'] < 100 and m['no_ask'] < 100)]
        current_positions = kalshi.get_positions()
        logger.info(f"postion {current_positions}")
        logger.info(f"Found {len(current_positions)} current positions.")
        
        # Extract tickers of current positions to filter
        held_tickers = {p['ticker'] for p in current_positions}
        
        logger.info(f"Found {len(viable_new_markets)} viable new active markets over the past {lookback_hours} hours.")
        
        for market in viable_new_markets:
            ticker = market['ticker']
            
            if ticker in held_tickers:
                logger.info(f"Skipping {ticker}, already held.")
                continue

            logger.info(f"Analyzing {ticker}: title: {market.get('title')}, yes_ask: {market.get('yes_ask')}, no_ask: {market.get('no_ask')}")
            
            # 4. Consult Grok
            recommendation = grok.analyze_market(market)
            
            rec_ticker = recommendation.get("ticker")
            explanation = recommendation.get("explanation")
            side = recommendation.get("side", "yes")   # <-- NEW: pass side from Grok
            
            if rec_ticker and rec_ticker == ticker:
                logger.info(f"Grok recommends BUY {side.upper()} on {ticker}. Reason: {explanation}")
                
                # Calculate order cost
                order_price = market.get(f"{side}_ask")
                order_count = 1
                order_cost = spending_tracker.calculate_order_cost(order_count, order_price)
                
                # Check spending limit
                can_place, reason = spending_tracker.can_place_order(order_cost, current_spending)
                
                if not can_place:
                    logger.warning(f"Skipping order for {ticker}: {reason}")
                    continue
                
                try:
                    logger.info(f"About to buy {order_count} {side} contract(s) for {ticker} at price {order_price} cents (cost: ${order_cost/100:.2f})")
                    
                    order_response = kalshi.create_market_order(
                        ticker,
                        side=side,
                        count=order_count,
                        price=order_price,
                        bot_identifier=bot_identifier
                    )
                    
                    # Update current spending after successful order
                    current_spending += order_cost
                    logger.info(f"Order placed successfully: {order_response}. Updated spending: ${current_spending/100:.2f}")
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
