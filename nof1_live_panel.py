"""
Streamlit live panel for NoF1.ai trading competition (fixed version)

This app pulls data from the NoF1.ai `/api/trades` endpoint, calculates key
performance metrics for the supported models and displays the results in a
refreshing dashboard.  The page automatically refreshes every minute to
keep the data up to date.

To deploy this app online so others can view it, sign up for a free
Streamlit Community Cloud account (https://streamlit.io/cloud), create a new
app and point it at this script.  The hosted app will provide a
shareable URL.
"""

import json
import time
from typing import Dict, Iterable, List

import pandas as pd
import requests
import streamlit as st

# Mapping of user‑friendly names to the model identifiers in the API
MODEL_MAP: Dict[str, str] = {
    "Deepseek": "deepseek-chat-v3.1",
    "Qwen3": "qwen3-max",
    "Claude": "claude-sonnet-4-5",
    "Grok4": "grok-4",
    "Gemini": "gemini-2.5-pro",
    "GPT5": "gpt-5",
}

TRADES_URL = "https://nof1.ai/api/trades"


def load_trades() -> List[dict]:
    """Fetch the latest trades from NoF1.ai and return a list of trade records."""
    try:
        response = requests.get(TRADES_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("trades", [])
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return []


def compute_metrics(trades: Iterable[dict], initial_capital: float = 10000.0) -> pd.DataFrame:
    """Compute performance metrics per model from raw trade records."""
    rows = []
    trades_list = list(trades)
    for name, model_id in MODEL_MAP.items():
        model_trades = [tr for tr in trades_list if tr.get("model_id") == model_id]
        total = len(model_trades)
        long_count = sum(1 for tr in model_trades if tr.get("side") == "long")
        short_count = sum(1 for tr in model_trades if tr.get("side") == "short")
        # ratio
        if short_count > 0:
            ratio = long_count / short_count
        elif long_count > 0:
            ratio = float("inf")
        else:
            ratio = float("nan")
        # profit and loss
        profits = [tr for tr in model_trades if tr.get("realized_net_pnl", 0) > 0]
        losses = [tr for tr in model_trades if tr.get("realized_net_pnl", 0) < 0]
        max_loss = min((tr.get("realized_net_pnl", 0) for tr in model_trades), default=0.0)
        max_profit = max((tr.get("realized_net_pnl", 0) for tr in model_trades), default=0.0)
        # max loss/traded amount
        loss_ratios = []
        for tr in model_trades:
            notional = abs(tr.get("entry_sz", 0) * tr.get("entry_price", 0))
            pnl = tr.get("realized_net_pnl", 0)
            if notional > 0 and pnl < 0:
                loss_ratios.append(abs(pnl) / notional)
        max_loss_ratio = max(loss_ratios) if loss_ratios else float("nan")
        hit_rate = (len(profits) / total * 100) if total > 0 else float("nan")
        # average holding time
        hold_times = [(tr.get("exit_time") - tr.get("entry_time")) / 60.0
                      for tr in model_trades
                      if tr.get("exit_time") is not None and tr.get("entry_time") is not None]
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else float("nan")
        # max notional
        notionals = [abs(tr.get("entry_sz", 0) * tr.get("entry_price", 0)) for tr in model_trades]
        max_notional = max(notionals) if notionals else float("nan")
        # leverage
        leverages = [tr.get("leverage", float("nan")) for tr in model_trades]
        max_leverage = max(leverages) if leverages else float("nan")
        avg_leverage = sum(leverages) / len(leverages) if leverages else float("nan")
        rows.append({
            "AI Model": name,
            "Total orders": total,
            "Long count": long_count,
            "Short count": short_count,
            "Long/Short ratio": ratio,
            "Number of profit orders": len(profits),
            "Number of loss orders": len(losses),
            "Max loss ($)": max_loss,
            "Max loss/traded amount": max_loss_ratio,
            "Max profit ($)": max_profit,
            "Hit rate (%)": hit_rate,
            "Average Holding time (min)": avg_hold,
            "Max notional value per position ($)": max_notional,
            "Max leverage": max_leverage,
            "Average leverage": avg_leverage,
            "Initial capital ($)": initial_capital,
        })
    df = pd.DataFrame(rows)
    df = df.replace([float("inf"), float("nan")], ["∞", "N/A"])
    # Transpose: metrics as rows, models as columns
    df = df.set_index("AI Model").T
    return df


def main():
    st.title("NoF1.ai Live Model Performance Panel")
    st.write("This dashboard reads completed trades from the NoF1.ai and updates every minute.")

    # Display the last refresh time
    st.write(f"Last refresh: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    trades = load_trades()
    if trades:
        df = compute_metrics(trades)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No data available.")

    # Add a meta tag to refresh the page every 60 seconds
    st.markdown("""
    <meta http-equiv="refresh" content="60">
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
