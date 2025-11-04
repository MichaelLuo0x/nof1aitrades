# nof1_live_panel.py

"""
Streamlit live panel for NoF1.ai trading competition
- Fetches trades from NoF1.ai
- Computes per-model metrics
- Transposes output: metrics as rows, models as columns
- Colored column headers per model (clean, aligned)
- KPI badges and auto-refresh

Safe, readable, and ready to run.
"""

import time
from typing import Dict, Iterable, List

import pandas as pd
import requests
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
    """
    Compute per-model metrics.
    Returns a transposed DataFrame with metrics as rows and models as columns.
    """
    rows = []
    trades_list = list(trades)

    for name, model_id in MODEL_MAP.items():
        model_trades = [tr for tr in trades_list if tr.get("model_id") == model_id]

        total = len(model_trades)
        long_count = sum(1 for tr in model_trades if tr.get("side") == "long")
        short_count = sum(1 for tr in model_trades if tr.get("side") == "short")

        # Long/Short ratio
        if short_count > 0:
            ratio = long_count / short_count
        elif long_count > 0:
            ratio = float("inf")
        else:
            ratio = float("nan")

        # Profit and loss counts and extrema (use realized_net_pnl)
        profits = [tr for tr in model_trades if tr.get("realized_net_pnl", 0) > 0]
        losses = [tr for tr in model_trades if tr.get("realized_net_pnl", 0) < 0]
        max_loss = min((tr.get("realized_net_pnl", 0) for tr in model_trades), default=0.0)
        max_profit = max((tr.get("realized_net_pnl", 0) for tr in model_trades), default=0.0)

        # Max loss / traded amount (severity)
        loss_ratios = []
        for tr in model_trades:
            notional = abs(tr.get("entry_sz", 0) * tr.get("entry_price", 0))
            pnl = tr.get("realized_net_pnl", 0)
            if notional > 0 and pnl < 0:
                loss_ratios.append(abs(pnl) / notional)
        max_loss_ratio = max(loss_ratios) if loss_ratios else float("nan")

        # Hit rate
        hit_rate = (len(profits) / total * 100) if total > 0 else float("nan")

        # Average holding time (minutes)
        hold_times = [
            (tr.get("exit_time") - tr.get("entry_time")) / 60.0
            for tr in model_trades
            if tr.get("exit_time") is not None and tr.get("entry_time") is not None
        ]
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else float("nan")

        # Max entry notional
        notionals = [abs(tr.get("entry_sz", 0) * tr.get("entry_price", 0)) for tr in model_trades]
        max_notional = max(notionals) if notionals else float("nan")

        # Leverage (filter None/NaN)
        leverages_raw = [tr.get("leverage") for tr in model_trades]
        leverages = [lv for lv in leverages_raw if isinstance(lv, (int, float)) and not pd.isna(lv)]
        if leverages:
            max_leverage = max(leverages)
            avg_leverage = sum(leverages) / len(leverages)
        else:
            max_leverage = float("nan")
            avg_leverage = float("nan")

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
            "Max entry notional ($)": max_notional,
            "Max leverage": max_leverage,
            "Average leverage": avg_leverage,
            "Initial capital ($)": initial_capital,
        })

    df = pd.DataFrame(rows)
    # Replace inf/nan for display
    df = df.replace([float("inf"), float("nan")], ["∞", "N/A"])
    # Transpose: metrics as rows, models as columns
    df = df.set_index("AI Model").T
    df.index.name = "Metric"
    return df


def format_k(number) -> str:
    """Format numbers with separators or K/M suffix."""
    if isinstance(number, (int, float)):
        try:
            absn = abs(number)
            if absn >= 1_000_000:
                return f"{number/1_000_000:.2f}M"
            if absn >= 10_000:
                return f"{number/1_000:.1f}K"
            return f"{number:,.0f}"
        except Exception:
            return str(number)
    return str(number)


def model_css_class(col_name: str) -> str:
    """Map DataFrame column (friendly model name) to CSS class name."""
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
    Render df (metrics rows × model columns) as HTML with colored column headers.
    Uses colored header cells (not badges) for perfect alignment.
    """
    # Header
    header_cells = ["<th class='metric-header'>Metric</th>"]
    for col in df.columns:
        cls = model_css_class(col)
        header_cells.append(
            f"<th class='model-header {cls}'>{col}</th>"
        )
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    # Body
    body_rows = []
    for metric, row in df.iterrows():
        cells = [f"<td class='metric-cell'>{metric}</td>"]
        for col in df.columns:
            val = row[col]
            cells.append(f"<td class='value-cell'>{val}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <table class="perf-table">
      <thead>
        {header_html}
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def main():
    # Page config
    st.set_page_config(
        page_title="NoF1.ai Live Model Performance",
        page_icon="📈",
        layout="wide",
    )

    # CSS theme and palette (clean, aligned headers)
    st.markdown("""
    <style>
      .metric-badge {
        display:inline-block;
        padding:6px 10px;
        border-radius:12px;
        background:#f6f8fa;
        border:1px solid #e1e4e8;
        margin-right:8px;
        margin-bottom:8px;
        font-size:13px;
      }

      /* Table base */
      .perf-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
      }
      .perf-table thead th, .perf-table tbody td {
        padding: 8px 12px;
      }
      .perf-table tbody tr:nth-child(odd) { background-color: #fafbfc; }
      .perf-table tbody tr:hover { background-color: #eef3f8; }

      /* Metric column */
      .metric-header {
        text-align: left;
        background: #111827;
        color: #fff;
        font-weight: 600;
        white-space: nowrap;
      }
      .metric-cell {
        font-weight: 500;
        white-space: nowrap;
      }
      .value-cell {
        text-align: center;
      }

      /* Colored model header cells (ensures alignment) */
      .model-header {
        color: #fff;
        text-align: center;
        font-weight: 700;
        white-space: nowrap;
      }
      .model-header.Deepseek { background:#6f42c1; } /* purple */
      .model-header.Qwen3   { background:#0366d6; } /* blue */
      .model-header.Claude  { background:#d73a49; } /* red */
      .model-header.Grok4   { background:#2da44e; } /* green */
      .model-header.Gemini  { background:#ff7b72; } /* coral */
      .model-header.GPT5    { background:#8250df; } /* violet */
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("## 📊 NoF1.ai Live Model Performance Panel")
    st.markdown("Analyze closed trades per model from the NoF1.ai competition. Auto-refreshes every 60s.")

    # KPIs
    last_refresh = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    trades = load_trades()

    col_info = st.columns([2, 1, 1, 1])
    with col_info[0]:
        st.markdown(f"<span class='metric-badge'>🕒 Last refresh: {last_refresh}</span>", unsafe_allow_html=True)

    total_trades = len(trades)
    by_model = pd.Series([tr.get("model_id") for tr in trades]).value_counts() if trades else pd.Series(dtype=int)
    with col_info[1]:
        st.markdown(f"<span class='metric-badge'>📦 Total trades: {format_k(total_trades)}</span>", unsafe_allow_html=True)
    with col_info[2]:
        st.markdown(f"<span class='metric-badge'>🤖 Models seen: {len(by_model)}</span>", unsafe_allow_html=True)
    with col_info[3]:
        top_model = by_model.idxmax() if len(by_model) else "N/A"
        st.markdown(f"<span class='metric-badge'>🏆 Most active: {top_model}</span>", unsafe_allow_html=True)

    # Data area
    if trades:
        df = compute_metrics(trades)
        st.markdown("#### Metrics (rows) × Models (columns)")
        render_colored_header_table(df)
    else:
        st.info("No data available right now. The panel will refresh automatically.")

    # Auto-refresh every 60 seconds
    st.markdown("""
    <meta http-equiv="refresh" content="60">
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
