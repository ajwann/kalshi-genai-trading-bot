import logging
from typing import Dict

import functions_framework
from dotenv import load_dotenv

from grok_client import GrokClient
from kalshi_client import KalshiClient
from spending_limit import SpendingLimitTracker
from utils import get_env_var, get_private_key

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _derive_series_ticker(event_ticker: str) -> str:
    """Best-effort extraction of series ticker from an event ticker.

    Kalshi convention is typically ``SERIES-DATE…`` so we strip the
    trailing date segment(s).  Returns the event ticker unchanged if
    we can't identify a date-like suffix.
    """
    parts = event_ticker.split("-")
    if len(parts) <= 1:
        return event_ticker

    # Walk from the right and drop parts that look like dates (digits,
    # short alphanumeric date codes like "26APR02", qualifier prefixes
    # like "T95000").
    prefix_parts = []
    for part in parts:
        has_digit = any(c.isdigit() for c in part)
        if has_digit and prefix_parts:
            break
        prefix_parts.append(part)

    return "-".join(prefix_parts) if prefix_parts else event_ticker


def build_settlement_rules(
    kalshi: KalshiClient,
    ticker: str,
    series_cache: Dict[str, Dict],
) -> str:
    """Fetch market detail + series and return a formatted rules block.

    Series lookups are cached in *series_cache* (keyed by series ticker)
    to avoid redundant API calls within a single bot run.
    """
    lines = ["--- Settlement Rules (authoritative for resolution) ---"]

    try:
        detail = kalshi.get_market(ticker)
    except Exception as exc:
        logger.warning("Could not fetch market detail for %s: %s", ticker, exc)
        return ""

    yes_sub = detail.get("yes_sub_title", "")
    no_sub = detail.get("no_sub_title", "")
    rules_primary = detail.get("rules_primary", "")
    rules_secondary = detail.get("rules_secondary", "")

    if yes_sub:
        lines.append(f"Yes outcome: {yes_sub}")
    if no_sub:
        lines.append(f"No outcome: {no_sub}")
    if rules_primary:
        lines.append(f"Primary rules: {rules_primary}")
    if rules_secondary:
        lines.append(f"Secondary rules: {rules_secondary}")

    # Attempt to enrich with series-level settlement sources
    event_ticker = detail.get("event_ticker", "")
    if event_ticker:
        series_ticker = _derive_series_ticker(event_ticker)
        series = series_cache.get(series_ticker)
        if series is None:
            try:
                series = kalshi.get_series(series_ticker)
                series_cache[series_ticker] = series
            except Exception as exc:
                logger.debug(
                    "Series lookup failed for %s (derived from %s): %s",
                    series_ticker,
                    event_ticker,
                    exc,
                )
                series = {}
                series_cache[series_ticker] = series

        sources = series.get("settlement_sources", [])
        if sources:
            source_strs = []
            for src in sources:
                name = src.get("name", "")
                url = src.get("url", "")
                source_strs.append(f"{name} ({url})" if url else name)
            lines.append(f"Settlement sources: {'; '.join(source_strs)}")

        frequency = series.get("frequency", "")
        if frequency:
            lines.append(f"Settlement frequency: {frequency}")

        contract_url = series.get("contract_url", "")
        if contract_url:
            lines.append(f"Contract filing: {contract_url}")

    lines.append("--- End Settlement Rules ---")
    return "\n".join(lines)


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
        max_spending_cents=spending_limit_cents,
    )

    # 3. Fetch Data
    try:
        # Get recent orders to calculate current spending
        try:
            all_orders = kalshi.get_orders(bot_identifier=bot_identifier)
            current_spending = spending_tracker.calculate_current_spending(all_orders)
            logger.info(
                "Current bot spending in last %s hours: $%.2f (limit: $%.2f)",
                spending_limit_period_hours,
                current_spending / 100,
                spending_limit_cents / 100,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch orders for spending calculation: {e}")
            current_spending = 0  # Assume no spending if we can't fetch

        new_markets = kalshi.get_active_markets(created_after_hours=lookback_hours)
        viable_new_markets = [
            market
            for market in new_markets
            if market["yes_ask"] < 100 and market["no_ask"] < 100
        ]
        current_positions = kalshi.get_positions()
        logger.info(f"position {current_positions}")
        logger.info(f"Found {len(current_positions)} current positions.")

        # Extract tickers of current positions to filter
        held_tickers = {position["ticker"] for position in current_positions}

        logger.info(
            "Found %s viable new active markets over the past %s hours.",
            len(viable_new_markets),
            lookback_hours,
        )

        series_cache: Dict[str, Dict] = {}

        for market in viable_new_markets:
            ticker = market["ticker"]

            if ticker in held_tickers:
                logger.info(f"Skipping {ticker}, already held.")
                continue

            logger.info(
                "Analyzing %s: title=%s, yes_ask=%s, no_ask=%s",
                ticker,
                market.get("title"),
                market.get("yes_ask"),
                market.get("no_ask"),
            )

            # 4. Fetch settlement rules and consult Grok
            settlement_rules = build_settlement_rules(kalshi, ticker, series_cache)
            if settlement_rules:
                logger.info("Injected settlement rules for %s", ticker)

            recommendation = grok.analyze_market(market, settlement_rules=settlement_rules)

            rec_ticker = recommendation.get("ticker")
            explanation = recommendation.get("explanation")
            side = recommendation.get("side", "yes")

            if rec_ticker and rec_ticker == ticker:
                logger.info(
                    "Grok recommends BUY %s on %s. Reason: %s",
                    side.upper(),
                    ticker,
                    explanation,
                )

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
                    logger.info(
                        "About to buy %s %s contract(s) for %s at price %s cents (cost: $%.2f)",
                        order_count,
                        side,
                        ticker,
                        order_price,
                        order_cost / 100,
                    )

                    order_response = kalshi.create_market_order(
                        ticker,
                        side=side,
                        count=order_count,
                        price=order_price,
                        bot_identifier=bot_identifier,
                    )

                    # Update current spending after successful order
                    current_spending += order_cost
                    logger.info(
                        "Order placed successfully: %s. Updated spending: $%.2f",
                        order_response,
                        current_spending / 100,
                    )
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
    load_dotenv()
    run_bot_logic()
