# nof1_live_panel.py

"""
Streamlit live panel for NoF1.ai trading competition
- Fetches trades LIVE from https://nof1.ai/api/trades (no local file used)
- Computes per-model metrics per spec
- Transposes output: metrics as rows, models as columns
- Colored column headers per model, responsive HTML table, alternating row shading
- Solid divider between metric groups
- All numeric values rounded to 4 decimals (∞ / N/A preserved)
NOTE: After a new trading season launches, I will enable live tracking again.
"""

from typing import Dict, Iterable, List, Tuple, Any
import math
import requests
import pandas as pd
import streamlit as st

# Friendly names -> model_id mapping
MODEL_MAP: Dict[str, str] = {
    "Deepseek": "deepseek-chat-v3.1",
    "Qwen3": "qwen3-max",
    "Claude": "claude-sonnet-4-5",
    "Grok4": "grok-4",
    "Gemini": "gemini-2.5-pro",
    "GPT5": "gpt-5",
}

TRADES_URL = "https://nof1.ai/api/trades"

# Static leverage assumptions supplied by competition organizers
STATIC_AVG_LEVERAGE: Dict[str, float] = {
    "Qwen3": 15.1,
    "Deepseek": 12.8,
    "Claude": 12.8,
    "Grok4": 12.1,
    "Gemini": 14.4,
    "GPT5": 16.7,
}


def load_trades() -> List[dict]:
    """Fetch the latest trades from NoF1.ai and return a list of trade records."""
    try:
        resp = requests.get(TRADES_URL, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        trades = payload.get("trades", [])
        # Some snapshots wrap each item like {"trades": {...}} — normalize just in case
        if trades and isinstance(trades[0], dict) and "trades" in trades[0]:
            trades = [t["trades"] for t in trades]
        return trades
    except Exception as exc:
        st.error(f"Could not load data from NoF1.ai: {exc}")
        return []


def _safe_num(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) else float("nan")


def compute_metrics(trades: Iterable[dict], initial_capital: float = 10000.0) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Compute per-model metrics.
    Returns:
      - DataFrame (metrics as rows, models as columns)
      - metrics_by_model: nested dict of raw metrics
    """
    trades_list = list(trades)
    metrics_by_model: Dict[str, Dict[str, Any]] = {}

    for name, model_id in MODEL_MAP.items():
        model_trades = [tr for tr in trades_list if tr.get("model_id") == model_id]
        total = len(model_trades)

        # Always include columns; if no trades, fill with NaNs/N/A later
        if total == 0:
            metrics_by_model[name] = {}
            continue

        # Long/short counts & ratio
        long_count = sum(1 for tr in model_trades if tr.get("side") == "long")
        short_count = sum(1 for tr in model_trades if tr.get("side") == "short")
        if short_count > 0:
            ratio = long_count / short_count
        elif long_count > 0:
            ratio = "∞"
        else:
            ratio = "N/A"

        # PnL vectors
        pnls = [_safe_num(tr.get("realized_net_pnl", 0.0)) for tr in model_trades]
        profits = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        num_profit = len(profits)
        num_loss = len(losses)
        hit_rate = (num_profit / total * 100.0) if total > 0 else float("nan")

        # Cumulative profit/loss & ROE
        total_profit = sum(profits) if profits else 0.0
        total_loss = sum(losses) if losses else 0.0
        roe = ((total_profit + total_loss) / initial_capital * 100.0) if initial_capital else float("nan")

        # Extremes
        max_profit = max(pnls) if pnls else 0.0
        max_loss = min(pnls) if pnls else 0.0

        # Average holding time (minutes) from numeric seconds
        hold_times = []
        for tr in model_trades:
            entry_t = tr.get("entry_time")
            exit_t = tr.get("exit_time")
            if isinstance(entry_t, (int, float)) and isinstance(exit_t, (int, float)):
                hold_times.append((exit_t - entry_t) / 60.0)
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else float("nan")

        # Max entry notional
        notionals = [abs(_safe_num(tr.get("entry_sz")) * _safe_num(tr.get("entry_price"))) for tr in model_trades]
        max_notional = max(notionals) if notionals else float("nan")

        # Total commission fees
        total_fees = sum(_safe_num(tr.get("total_commission_dollars", 0.0)) for tr in model_trades)

        metrics_by_model[name] = {
            "Total orders": total,
            "Long count": long_count,
            "Short count": short_count,
            "Long/Short ratio": ratio,
            "Number of profit orders": num_profit,
            "Number of loss orders": num_loss,
            "Hit rate (%)": hit_rate,
            "Cumulative profit ($)": total_profit,
            "Cumulative loss ($)": total_loss,
            "Return on equity (%)": roe,
            "Max profit ($)": max_profit,
            "Max loss ($)": max_loss,
            "Average Holding time (min)": avg_hold,
            "Max entry notional ($)": max_notional,
            "Total commission fees ($)": total_fees,
            "Initial capital ($)": initial_capital,
        }
        metrics_by_model[name]["Average leverage"] = STATIC_AVG_LEVERAGE.get(name, float("nan"))

    for name in MODEL_MAP:
        metrics_by_model.setdefault(name, {})
        metrics_by_model[name].setdefault("Average leverage", STATIC_AVG_LEVERAGE.get(name, float("nan")))

    # Desired order
    top_metrics = [
        "Total orders",
        "Long count",
        "Short count",
        "Long/Short ratio",
        "Number of profit orders",
        "Number of loss orders",
        "Hit rate (%)",
    ]
    bottom_metrics = [
        "Cumulative profit ($)",
        "Cumulative loss ($)",
        "Return on equity (%)",
        "Max profit ($)",
        "Max loss ($)",
        "Average Holding time (min)",
        "Max entry notional ($)",
        "Average leverage",
        "Total commission fees ($)",
        "Initial capital ($)",
    ]

    # Ensure all models present
    all_models = list(MODEL_MAP.keys())
    rows = top_metrics + bottom_metrics

    table: Dict[str, Dict[str, Any]] = {m: {} for m in rows}
    for metric in rows:
        for model in all_models:
            val = metrics_by_model.get(model, {}).get(metric, float("nan"))
            table[metric][model] = val

    df = pd.DataFrame.from_dict(table, orient="index")
    df.index.name = "Metric"
    return df, metrics_by_model


def model_css_class(col_name: str) -> str:
    return {
        "Deepseek": "Deepseek",
        "Qwen3": "Qwen3",
        "Claude": "Claude",
        "Grok4": "Grok4",
        "Gemini": "Gemini",
        "GPT5": "GPT5",
    }.get(col_name, "")


def render_colored_header_table(df: pd.DataFrame) -> None:
    """
    Render df (metrics rows × model columns) as HTML with color-coded column headers.
    Includes a solid horizontal divider between metric groups.
    Responsive (horizontal scroll) + alternating row shading.
    All numeric values rendered to exactly 4 decimals (∞/N/A preserved).
    """
    cols = list(df.columns)

    # Header
    header_cells = ["<th class='metric-header'>Metric</th>"]
    for col in cols:
        header_cells.append(f"<th class='model-header {model_css_class(col)}'>{col}</th>")
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    # Body
    top_count = 7  # number of top metrics
    body_rows = []
    for idx, (metric, row) in enumerate(df.iterrows(), start=1):
        # Divider row before bottom group
        if idx == top_count + 1:
            body_rows.append(f"<tr><td class='divider' colspan='{len(cols)+1}'></td></tr>")
        # metric name
        cells = [f"<td class='metric-cell'>{metric}</td>"]
        # values per model
        for col in cols:
            val = row[col]
            if isinstance(val, str):
                out = val  # '∞'/'N/A' or ratios that are already strings
            elif isinstance(val, (int, float)):
                if val == float("inf"):
                    out = "∞"
                elif math.isnan(val):
                    out = "N/A"
                else:
                    out = f"{val:.4f}"
            else:
                out = "N/A"
            cells.append(f"<td class='value-cell'>{out}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div class="table-wrap">
      <table class="perf-table">
        <thead>
          {header_html}
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="NoF1.ai Model Performance", page_icon="📈", layout="wide")
    st.markdown("## 📊 NoF1.ai Model Performance Panel")
    st.markdown("_Note: After a new trading season launches, live tracking of trades will resume._")

    # CSS: theme, colors, responsiveness, alternating rows, divider
    st.markdown("""
    <style>
      .table-wrap { width:100%; overflow-x:auto; }
      .perf-table { width:100%; border-collapse: separate; border-spacing:0; }
      .perf-table thead th, .perf-table tbody td { padding: 10px 12px; }
      .perf-table thead th { position: sticky; top: 0; z-index: 2; }
      .perf-table tbody tr:nth-child(odd) { background-color: #fafbfc; }
      .perf-table tbody tr:nth-child(even) { background-color: #ffffff; }
      .perf-table tbody tr:hover { background-color: #eef3f8; }

      .metric-header { text-align:left; background:#111827; color:#fff; font-weight:700; white-space:nowrap; }
      .metric-cell { font-weight:600; white-space:nowrap; }
      .value-cell { text-align:center; }

      /* Solid divider row */
      .divider { border-top: 3px solid #000; height:0; padding:0; }

      /* Colored model header cells (your palette) */
      .model-header { color:#fff; text-align:center; font-weight:800; white-space:nowrap; }
      .model-header.Deepseek { background:#0366d6 !important; } /* blue */
      .model-header.Qwen3   { background:#6f42c1 !important; } /* purple */
      .model-header.Claude  { background:#d73a49 !important; } /* orange */
      .model-header.Grok4   { background:#000000 !important; } /* black */
      .model-header.Gemini  { background:#007FFF !important; } /* azure */
      .model-header.GPT5    { background:#2da44e !important; } /* green */
    </style>
    """, unsafe_allow_html=True)

    trades = load_trades()
    if not trades:
        st.info("No trade data available right now.")
        return

    df, _ = compute_metrics(trades)
    render_colored_header_table(df)


if __name__ == "__main__":
    main()
