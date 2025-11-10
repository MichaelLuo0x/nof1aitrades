"""Streamlit dashboard for Hyperliquid trade post-mortem."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from hyperliquid_fetch import (
    MODEL_ADDRESSES,
    default_csv_path_for_model,
    fetch_hyperliquid_trades,
    model_slug,
    write_trades_to_csv,
)
from hyperliquid_trade_picture import MODEL_FILES, process_trade_picture

MODEL_LABELS: Dict[str, str] = {
    "qwen3": "Qwen3 Max",
    "deepseek": "DeepSeek Chat V3.1",
    "gemini": "Gemini 2.5 Pro",
    "claude": "Claude Sonnet 4.5",
    "grok4": "Grok4",
    "gpt5": "GPT-5",
}

MODEL_HEADER_COLORS: Dict[str, str] = {
    "Qwen3 Max": "#8B5CF6",
    "DeepSeek Chat V3.1": "#3B82F6",
    "Claude Sonnet 4.5": "#F97316",
    "Gemini 2.5 Pro": "#60A5FA",
    "Grok4": "#1F1F1F",
    "GPT-5": "#4CAF87",
}


def ideal_text_color(hex_color: str) -> str:
    hex_color = (hex_color or "").lstrip("#")
    if len(hex_color) != 6:
        return "#FFFFFF"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.65 else "#FFFFFF"


def model_color(slug_or_label: str) -> str:
    label = MODEL_LABELS.get(slug_or_label, slug_or_label)
    return MODEL_HEADER_COLORS.get(label, "#2ca02c")


def model_color_scale() -> alt.Scale:
    domain = list(MODEL_HEADER_COLORS.keys())
    range_vals = [MODEL_HEADER_COLORS[label] for label in domain]
    return alt.Scale(domain=domain, range=range_vals)

SUMMARY_METRICS = [
    "Net PnL ($)",
    "Total Fees ($)",
    "Total Long Trades",
    "Total Short Trades",
    "Long/Short Ratio",
    "Max Concurrent Trades",
    "Max Gain ($)",
    "Max Loss ($)",
    "Win Rate (%)",
]


def refresh_model_data(slugs: List[str], limit: int = 1000) -> None:
    for slug in slugs:
        address = MODEL_ADDRESSES[slug]
        raw_csv = Path(default_csv_path_for_model(slug))
        st.write(f"Fetching {MODEL_LABELS.get(slug, slug)} trades...")
        trades = fetch_hyperliquid_trades(address=address, limit=limit)
        write_trades_to_csv(trades, raw_csv)
        input_path, output_path = MODEL_FILES[slug]
        process_trade_picture(input_path, output_path, show_summary=False, label=slug)


def load_trade_picture(slug: str) -> pd.DataFrame:
    _, output_path = MODEL_FILES[slug]
    path = Path(output_path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    df["closedPnl"] = pd.to_numeric(df.get("closedPnl"), errors="coerce").fillna(0.0)
    df["fee"] = pd.to_numeric(df.get("fee"), errors="coerce").fillna(0.0)
    df["net_pnl"] = df["closedPnl"] - df["fee"]
    df["px"] = pd.to_numeric(df.get("px"), errors="coerce").fillna(0.0)
    df["sz"] = pd.to_numeric(df.get("sz"), errors="coerce").fillna(0.0)
    df["startPosition"] = pd.to_numeric(df.get("startPosition"), errors="coerce").fillna(0.0)
    df["position_after"] = pd.to_numeric(df.get("Position size"), errors="coerce").fillna(0.0)
    df["time_ts"] = pd.to_datetime(df.get("time"), errors="coerce")
    df["notional"] = (df["startPosition"].abs()) * df["px"]
    df["is_long_order"] = df["dir"].str.contains("Long", case=False, na=False)
    df["is_short_order"] = df["dir"].str.contains("Short", case=False, na=False)
    if "symbol" in df.columns:
        if "coin" in df.columns:
            df["symbol"] = df["symbol"].fillna(df["coin"])
    else:
        df["symbol"] = df["coin"] if "coin" in df.columns else ""
    if "oid" in df.columns:
        oid_numeric = pd.to_numeric(df["oid"], errors="coerce")
        df["oid_key"] = oid_numeric.apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
    else:
        df["oid_key"] = ""
    mask = df["oid_key"].isin({"", "nan", "None"})
    df.loc[mask, "oid_key"] = df.index[mask].map(lambda idx: f"row_{idx}")

    trade_side_map = (
        df.groupby("trade_id")["dir"]
        .first()
        .str.contains("Long", case=False, na=False)
        .map({True: "Long", False: "Short"})
    )
    df["side_type"] = df["trade_id"].map(trade_side_map).fillna("Long")
    return df


def trade_level_view(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_id",
        "net_pnl",
        "total_fee",
        "order_count",
        "side",
        "max_notional",
        "start_time",
        "end_time",
        "symbol",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        df.groupby("trade_id")
        .agg(
            net_pnl=("net_pnl", "sum"),
            total_fee=("fee", "sum"),
            order_count=("oid_key", "nunique"),
            side=("side_type", "first"),
            max_notional=("notional", "max"),
            start_time=("time_ts", "min"),
            end_time=("time_ts", "max"),
            symbol=("symbol", "first"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values("start_time")
    return grouped


def summarize_model(df: pd.DataFrame, slug: str) -> Dict[str, float]:
    grouped = trade_level_view(df)
    if grouped.empty:
        return {
            "Model": MODEL_LABELS.get(slug, slug),
            "Net PnL ($)": 0.0,
            "Total Fees ($)": 0.0,
            "Total Long Trades": 0,
            "Total Short Trades": 0,
            "Long/Short Ratio": "N/A",
            "Max Concurrent Trades": 0,
            "Max Gain ($)": 0.0,
            "Max Loss ($)": 0.0,
            "Win Rate (%)": 0.0,
        }

    total_long = int((grouped["side"] == "Long").sum())
    total_short = int((grouped["side"] == "Short").sum())
    ratio = "∞" if total_short == 0 and total_long > 0 else f"{total_long / total_short:.2f}" if total_short else "N/A"
    wins = (grouped["net_pnl"] > 0).sum()
    win_rate = (wins / len(grouped) * 100.0) if len(grouped) else 0.0
    max_concurrent, _ = concurrency_stats(grouped)

    return {
        "Model": MODEL_LABELS.get(slug, slug),
        "Net PnL ($)": grouped["net_pnl"].sum(),
        "Total Fees ($)": grouped["total_fee"].sum(),
        "Total Long Trades": total_long,
        "Total Short Trades": total_short,
        "Long/Short Ratio": ratio,
        "Max Concurrent Trades": max_concurrent,
        "Max Gain ($)": grouped["net_pnl"].max() if not grouped.empty else 0.0,
        "Max Loss ($)": grouped["net_pnl"].min() if not grouped.empty else 0.0,
        "Win Rate (%)": win_rate,
    }


def multi_model_equity_chart(model_dfs: Dict[str, pd.DataFrame]) -> alt.Chart:
    records: List[pd.DataFrame] = []
    for slug, df in model_dfs.items():
        trade_df = trade_level_view(df)
        if trade_df.empty:
            continue
        trade_df = trade_df.copy()
        trade_df["equity"] = trade_df["net_pnl"].cumsum()
        trade_df["Model"] = MODEL_LABELS.get(slug, slug)
        records.append(trade_df[["start_time", "equity", "Model"]])

    if not records:
        return alt.Chart(pd.DataFrame({"start_time": [], "equity": [], "Model": []})).mark_line()

    data = pd.concat(records, ignore_index=True)
    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("start_time:T", title="Trade Start Time"),
            y=alt.Y("equity:Q", title="Cumulative Net PnL ($)"),
            color=alt.Color("Model:N", title="Model", scale=model_color_scale()),
            tooltip=["Model", "start_time:T", alt.Tooltip("equity:Q", format=",.2f")],
        )
        .properties(height=300)
    )


def drawdown_chart(df: pd.DataFrame, label: str) -> alt.Chart:
    trade_df = trade_level_view(df)
    if trade_df.empty:
        return alt.Chart(pd.DataFrame({"start_time": [], "metric": [], "value": []})).mark_line()

    trade_df = trade_df.copy()
    trade_df["equity"] = trade_df["net_pnl"].cumsum()
    trade_df["running_max"] = trade_df["equity"].cummax()
    trade_df["drawdown"] = trade_df["equity"] - trade_df["running_max"]

    base_color = model_color(label)
    line = (
        alt.Chart(trade_df)
        .mark_line(color=base_color)
        .encode(x=alt.X("start_time:T", title="Trade Start Time"), y=alt.Y("equity:Q", title="Equity ($)"))
    )
    area = (
        alt.Chart(trade_df)
        .mark_area(color=base_color, opacity=0.25)
        .encode(x="start_time:T", y=alt.Y("drawdown:Q", title="Drawdown ($)", axis=alt.Axis(titleColor=base_color)))
    )
    return (
        alt.layer(line, area)
        .resolve_scale(y="independent")
        .properties(height=300, title=f"Equity & Drawdown - {label}")
    )


def pnl_fee_scatter_chart(df: pd.DataFrame) -> alt.Chart:
    trade_df = trade_level_view(df)
    if trade_df.empty:
        return alt.Chart(pd.DataFrame({"net_pnl": [], "total_fee": [], "max_notional": []})).mark_circle()

    return (
        alt.Chart(trade_df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X("total_fee:Q", title="Total Fees ($)"),
            y=alt.Y("net_pnl:Q", title="Net PnL ($)"),
            size=alt.Size("max_notional:Q", title="Notional", scale=alt.Scale(range=[50, 600])),
            color=alt.condition("datum.net_pnl >= 0", alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=[
                "trade_id",
                alt.Tooltip("total_fee:Q", title="Fees", format=",.2f"),
                alt.Tooltip("net_pnl:Q", title="Net PnL", format=",.2f"),
                alt.Tooltip("max_notional:Q", title="Max Notional", format=",.2f"),
            ],
        )
        .properties(height=300, title="Net PnL vs. Fees")
    )


def session_heatmap_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty or "time_ts" not in df:
        return alt.Chart(pd.DataFrame({"weekday": [], "hour": [], "net_pnl": []})).mark_rect()

    temp = df.dropna(subset=["time_ts"]).copy()
    if temp.empty:
        return alt.Chart(pd.DataFrame({"weekday": [], "hour": [], "net_pnl": []})).mark_rect()

    temp["weekday"] = temp["time_ts"].dt.day_name().str[:3]
    temp["hour"] = temp["time_ts"].dt.hour
    grouped = temp.groupby(["weekday", "hour"]).agg(net_pnl=("net_pnl", "sum")).reset_index()
    if grouped.empty:
        return alt.Chart(pd.DataFrame({"weekday": [], "hour": [], "net_pnl": []})).mark_rect()

    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grouped["weekday"] = pd.Categorical(grouped["weekday"], categories=weekday_order, ordered=True)

    return (
        alt.Chart(grouped)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour of Day"),
            y=alt.Y("weekday:O", title="Weekday"),
            color=alt.Color("net_pnl:Q", title="Net PnL ($)", scale=alt.Scale(scheme="redblue", domainMid=0)),
            tooltip=["weekday", "hour", alt.Tooltip("net_pnl:Q", format=",.2f")],
        )
        .properties(height=250, title="PnL by Weekday & Hour")
    )


def symbol_contribution_chart(df: pd.DataFrame) -> alt.Chart:
    trade_df = trade_level_view(df)
    if trade_df.empty:
        return alt.Chart(pd.DataFrame({"symbol": [], "net_pnl": []})).mark_bar()

    temp = trade_df.copy()
    temp["symbol"] = temp["symbol"].fillna("Unknown")
    contrib = temp.groupby("symbol")["net_pnl"].sum().reset_index(name="net_pnl")
    contrib = contrib.sort_values("net_pnl", key=lambda s: s.abs(), ascending=False).head(15)

    return (
        alt.Chart(contrib)
        .mark_bar()
        .encode(
            y=alt.Y("symbol:N", sort="-x", title="Symbol"),
            x=alt.X("net_pnl:Q", title="Net PnL ($)"),
            color=alt.condition("datum.net_pnl >= 0", alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=["symbol", alt.Tooltip("net_pnl:Q", format=",.2f")],
        )
        .properties(height=300, title="Symbol Contribution")
    )
def model_net_pnl_chart(summary_df: pd.DataFrame) -> alt.Chart:
    if summary_df.empty:
        return alt.Chart(pd.DataFrame({"Model": [], "Net PnL ($)": []})).mark_bar()

    chart_df = summary_df.sort_values("Net PnL ($)", ascending=False).copy()
    chart_df["Model"] = chart_df["Model"].astype(str)
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Net PnL ($):Q", title="Net PnL ($)"),
            y=alt.Y("Model:N", sort="-x", title="Model"),
            color=alt.Color("Model:N", scale=model_color_scale(), legend=None),
            tooltip=["Model", alt.Tooltip("Net PnL ($):Q", format=",.2f")],
        )
        .properties(height=250, title="Net PnL by Model")
    )


def concurrency_stats(trade_df: pd.DataFrame) -> Tuple[int, float]:
    events: List[Tuple[pd.Timestamp, int]] = []
    for _, row in trade_df.dropna(subset=["start_time"]).iterrows():
        start = row["start_time"]
        end = row["end_time"]
        if pd.isna(end) or end < start:
            end = start
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return 0, 0.0

    events.sort(key=lambda item: (item[0], 0 if item[1] > 0 else 1))
    active = 0
    max_active = 0
    weighted_seconds = 0.0
    total_seconds = 0.0
    prev_time = events[0][0]
    for time, delta in events:
        delta_seconds = (time - prev_time).total_seconds()
        if delta_seconds > 0:
            weighted_seconds += active * delta_seconds
            total_seconds += delta_seconds
        active += delta
        max_active = max(max_active, active)
        prev_time = time

    avg_active = weighted_seconds / total_seconds if total_seconds > 0 else float(max_active)
    return max_active, avg_active


def trade_order_pnl_chart(df: pd.DataFrame, trade_id: str) -> alt.Chart:
    subset = df[df["trade_id"] == trade_id].copy()
    if subset.empty:
        return alt.Chart(pd.DataFrame({"time_ts": [], "net_pnl": []})).mark_point()

    order_df = (
        subset.groupby("oid_key")
        .agg(
            net_pnl=("net_pnl", "sum"),
            fee=("fee", "sum"),
            time_ts=("time_ts", "max"),
            direction=("dir", "first"),
        )
        .reset_index()
    )
    order_df = order_df.dropna(subset=["time_ts"])
    if order_df.empty:
        return alt.Chart(pd.DataFrame({"time_ts": [], "net_pnl": []})).mark_point()

    return (
        alt.Chart(order_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("time_ts:T", title="Time"),
            y=alt.Y("net_pnl:Q", title="Order Net PnL ($)"),
            color=alt.condition("datum.net_pnl >= 0", alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=[
                "oid_key",
                "direction",
                alt.Tooltip("net_pnl:Q", title="Net PnL", format=",.2f"),
                alt.Tooltip("fee:Q", title="Fee", format=",.2f"),
            ],
        )
        .properties(height=250, title="Order-Level PnL")
    )


def model_symbol_mix_chart(model_dfs: Dict[str, pd.DataFrame], top_symbols: int = 8) -> alt.Chart:
    rows: List[Dict[str, Any]] = []
    for slug, df in model_dfs.items():
        trade_df = trade_level_view(df)
        if trade_df.empty:
            continue
        label = MODEL_LABELS.get(slug, slug)
        symbols = trade_df["symbol"].fillna("Unknown")
        counts = symbols.value_counts()
        for symbol, count in counts.items():
            rows.append({"Model": label, "symbol": symbol, "count": int(count)})

    if not rows:
        return alt.Chart(pd.DataFrame({"Model": [], "symbol": [], "count": []})).mark_bar()

    data = pd.DataFrame(rows)
    totals = data.groupby("symbol")["count"].sum().sort_values(ascending=False)
    keep = set(totals.head(top_symbols).index)
    data.loc[~data["symbol"].isin(keep), "symbol"] = "Other"
    aggregated = data.groupby(["Model", "symbol"], as_index=False)["count"].sum()
    model_order = list(dict.fromkeys(aggregated["Model"].tolist()))

    return (
        alt.Chart(aggregated)
        .mark_bar()
        .encode(
            y=alt.Y("Model:N", sort=model_order, title="Model"),
            x=alt.X("count:Q", stack="normalize", title="Share of Trades"),
            color=alt.Color("symbol:N", title="Symbol"),
            tooltip=["Model", "symbol", alt.Tooltip("count:Q", title="Trades")],
        )
        .properties(height=280, title="Coin Mix per Model")
    )


def trade_pnl_chart(df: pd.DataFrame) -> alt.Chart:
    grouped = (
        df.groupby("trade_id")
        .agg(net_pnl=("net_pnl", "sum"))
        .reset_index()
        .sort_values("net_pnl", ascending=False)
    )
    if grouped.empty:
        return alt.Chart(pd.DataFrame({"trade_id": [], "net_pnl": []})).mark_bar()

    grouped["trade_idx"] = grouped.index + 1
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("trade_idx:O", title="Trade Group #"),
            y=alt.Y("net_pnl:Q", title="Net PnL ($)"),
            color=alt.condition("datum.net_pnl >= 0", alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=["trade_id", "net_pnl"],
        )
        .properties(height=300)
    )
    return chart


def trade_order_count_chart(df: pd.DataFrame) -> alt.Chart:
    aggregated = (
        df.groupby(["trade_id", "side_type"], as_index=False)
        .agg(order_count=("oid_key", "nunique"))
        .sort_values("trade_id")
    )
    if aggregated.empty:
        return alt.Chart(pd.DataFrame({"trade_id": [], "order_count": []})).mark_bar()

    aggregated["trade_idx"] = aggregated.index + 1
    color_scale = alt.Scale(domain=["Long", "Short"], range=["#2ca02c", "#d62728"])
    return (
        alt.Chart(aggregated)
        .mark_bar()
        .encode(
            x=alt.X("trade_idx:O", title="Trade Group #"),
            y=alt.Y("order_count:Q", title="Orders (unique oid)", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("side_type:N", title="Direction", scale=color_scale),
            tooltip=["trade_id", "order_count", "side_type"],
        )
        .properties(height=300)
    )


def fee_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    grouped = df.groupby("trade_id").agg(fees=("fee", "sum")).reset_index()
    return (
        alt.Chart(grouped)
        .mark_bar(color="#6f42c1")
        .encode(
            x=alt.X("fees:Q", bin=alt.Bin(maxbins=30), title="Fees per Trade ($)"),
            y=alt.Y("count():Q", title="Number of Trades"),
        )
        .properties(height=250)
    )


def holding_time_chart(df: pd.DataFrame) -> alt.Chart:
    grouped = (
        df.groupby("trade_id")
        .agg(min_time=("time_ts", "min"), max_time=("time_ts", "max"))
        .reset_index()
    )
    grouped = grouped.dropna(subset=["min_time", "max_time"])
    if grouped.empty:
        return alt.Chart(pd.DataFrame({"duration_min": []})).mark_bar()

    grouped["duration_min"] = (
        (grouped["max_time"] - grouped["min_time"]).dt.total_seconds() / 60.0
    )
    grouped = grouped.dropna(subset=["duration_min"])
    if grouped.empty:
        return alt.Chart(pd.DataFrame({"duration_min": []})).mark_bar()

    return (
        alt.Chart(grouped)
        .mark_bar(color="#ff7f0e")
        .encode(
            x=alt.X(
                "duration_min:Q",
                bin=alt.Bin(maxbins=30),
                title="Holding Time (min)",
            ),
            y=alt.Y("count():Q", title="Number of Trades"),
        )
        .properties(height=250)
    )


def multi_model_holding_chart(model_dfs: Dict[str, pd.DataFrame]) -> alt.Chart:
    rows: List[Dict[str, Any]] = []
    for slug, df in model_dfs.items():
        trade_df = trade_level_view(df)
        if trade_df.empty:
            continue
        label = MODEL_LABELS.get(slug, slug)
        durations = (trade_df["end_time"] - trade_df["start_time"]).dt.total_seconds() / 60.0
        durations = durations.dropna()
        for value in durations:
            rows.append({"Model": label, "duration_min": value})

    if not rows:
        return alt.Chart(pd.DataFrame({"Model": [], "duration_min": []})).mark_boxplot()

    data = pd.DataFrame(rows)
    data["Model"] = pd.Categorical(
        data["Model"],
        categories=[MODEL_LABELS.get(slug, slug) for slug in MODEL_LABELS if MODEL_LABELS.get(slug, slug) in data["Model"].unique()],
        ordered=True,
    )

    return (
        alt.Chart(data)
        .mark_boxplot(extent="min-max")
        .encode(
            y=alt.Y("Model:N", title="Model"),
            x=alt.X("duration_min:Q", title="Holding Time (min)", scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N", scale=model_color_scale(), legend=None),
            tooltip=[
                "Model",
                alt.Tooltip("min(duration_min):Q", title="Min", format=",.1f"),
                alt.Tooltip("median(duration_min):Q", title="Median", format=",.1f"),
                alt.Tooltip("max(duration_min):Q", title="Max", format=",.1f"),
            ],
        )
        .properties(height=260, title="Holding Time by Model (Box Plot)")
    )


def trade_summary(df: pd.DataFrame, trade_id: str) -> Dict[str, float]:
    subset = df[df["trade_id"] == trade_id].copy()
    if subset.empty:
        return {}
    grouped = (
        subset.groupby("trade_id")
        .agg(
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("closedPnl", "sum"),
            total_fee=("fee", "sum"),
            orders=("oid_key", "nunique"),
            max_position=("position_after", "max"),
            min_time=("time_ts", "min"),
            max_time=("time_ts", "max"),
        )
        .reset_index()
    )
    row = grouped.iloc[0]
    duration = (
        (row["max_time"] - row["min_time"]).total_seconds() / 60.0
        if pd.notnull(row["min_time"]) and pd.notnull(row["max_time"])
        else 0.0
    )
    return {
        "Net PnL": row["net_pnl"],
        "Gross PnL": row["gross_pnl"],
        "Total Fees": row["total_fee"],
        "Orders": row["orders"],
        "Max Position": row["max_position"],
        "Duration (min)": duration,
    }


def trade_detail_table(df: pd.DataFrame, trade_id: str) -> pd.DataFrame:
    subset = df[df["trade_id"] == trade_id].copy()
    if subset.empty:
        return pd.DataFrame()
    desired_columns = [
        "time",
        "symbol",
        "dir",
        "startPosition",
        "position_after",
        "px",
        "closedPnl",
        "fee",
        "net_pnl",
        "oid_key",
    ]
    available = [col for col in desired_columns if col in subset.columns]
    subset = subset[available].rename(
        columns={
            "time": "Time",
            "symbol": "Symbol",
            "dir": "Direction",
            "startPosition": "Start Position",
            "position_after": "Position Size",
            "px": "Price",
            "closedPnl": "Closed PnL",
            "fee": "Fee",
            "net_pnl": "Net PnL",
            "oid_key": "Order ID",
        }
    )
    return subset


def trade_timeline_chart(df: pd.DataFrame, trade_id: str) -> alt.Chart:
    subset = df[df["trade_id"] == trade_id].copy()
    if subset.empty:
        return alt.Chart(pd.DataFrame({"time": [], "position": []})).mark_line()

    def pick_final(group: pd.DataFrame) -> pd.DataFrame:
        direction = str(group.iloc[0]["dir"]).lower()
        if "open long" in direction:
            idx = group["position_after"].idxmax()
        elif "close long" in direction:
            idx = group["position_after"].idxmin()
        elif "open short" in direction:
            idx = group["position_after"].idxmin()
        elif "close short" in direction:
            idx = group["position_after"].idxmax()
        else:
            idx = group.index[-1]
        return group.loc[[idx]]

    subset = (
        subset.sort_values("row_id")
        .groupby("oid_key", group_keys=False)
        .apply(pick_final, include_groups=False)
        .sort_values("time_ts")
    )
    if subset.empty:
        return alt.Chart(pd.DataFrame({"time": [], "position": []})).mark_line()

    rows = []
    prev = 0.0
    for _, row in subset.iterrows():
        rows.append(
            {"time_ts": row["time_ts"], "position": prev, "px": None, "time": row["time"], "label": ""}
        )
        prev = row["position_after"]
        rows.append(
            {"time_ts": row["time_ts"], "position": prev, "px": row["px"], "time": row["time"], "label": f"{row['px']:.2f}"}
        )

    chart_df = pd.DataFrame(rows)
    line = (
        alt.Chart(chart_df)
        .mark_line(color="#2ca02c")
        .encode(
            x=alt.X("time_ts:T", title="Time"),
            y=alt.Y("position:Q", title="Position Size"),
            tooltip=["time", "position", "px"],
        )
    )
    points = (
        alt.Chart(chart_df[chart_df["px"].notna()])
        .mark_point(color="#1f77b4", size=60)
        .encode(x="time_ts:T", y="position:Q")
    )
    labels = (
        alt.Chart(chart_df[chart_df["px"].notna()])
        .mark_text(dx=5, dy=-8, fontSize=10, color="#1f77b4")
        .encode(x="time_ts:T", y="position:Q", text="label")
    )
    return (line + points + labels).properties(height=300)


def main() -> None:
    st.set_page_config(page_title="Hyperliquid Trade Analysis", layout="wide")
    st.title("Hyperliquid Trade Post-Mortem")

    st.sidebar.header("Controls")
    refresh_options = ["ALL"] + list(MODEL_LABELS.keys())
    selected_models = st.sidebar.multiselect(
        "Select models",
        options=list(MODEL_LABELS.keys()),
        default=list(MODEL_LABELS.keys()),
        format_func=lambda slug: MODEL_LABELS.get(slug, slug),
    )
    refresh_models = st.sidebar.multiselect(
        "Refresh data for models",
        options=refresh_options,
        default=[],
        format_func=lambda slug: "All Models" if slug == "ALL" else MODEL_LABELS.get(slug, slug),
    )
    limit = st.sidebar.number_input("Fetch limit per model", min_value=200, max_value=5000, value=1000, step=100)

    if st.sidebar.button("Run data refresh") and refresh_models:
        if "ALL" in refresh_models:
            target_models = list(MODEL_LABELS.keys())
        else:
            target_models = refresh_models
        with st.spinner("Refreshing data from Hyperliquid..."):
            refresh_model_data(target_models, limit=limit)
        st.success("Data refresh completed.")

    if not selected_models:
        st.info("Select at least one model to display results.")
        return

    dfs = {slug: load_trade_picture(slug) for slug in selected_models}
    summaries = [summarize_model(df, slug) for slug, df in dfs.items()]

    st.subheader("Performance Summary")
    summary_df = pd.DataFrame(summaries)
    pivot_df = summary_df.set_index("Model").T.loc[SUMMARY_METRICS]
    format_dict = {
        "Net PnL ($)": "{:,.2f}",
        "Total Fees ($)": "{:,.2f}",
        "Max Gain ($)": "{:,.2f}",
        "Max Loss ($)": "{:,.2f}",
        "Win Rate (%)": "{:,.2f}",
    }
    styler = pivot_df.style.format(format_dict)
    base_styles = [
        {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%"), ("font-size", "0.95rem")]},
        {"selector": "th.col_heading", "props": [("text-align", "center"), ("padding", "8px 10px"), ("border", "1px solid #e5e7eb")]},
        {"selector": "th.row_heading", "props": [("text-align", "left"), ("padding", "8px 10px"), ("font-weight", "600"), ("border", "1px solid #e5e7eb")]},
        {"selector": "td", "props": [("text-align", "right"), ("padding", "6px 10px"), ("border", "1px solid #f0f0f0")]},
    ]
    header_styles = []
    for idx, col in enumerate(pivot_df.columns):
        color = MODEL_HEADER_COLORS.get(col)
        if not color:
            continue
        text_color = ideal_text_color(color)
        header_styles.append(
            {
                "selector": f"th.col_heading.level0.col{idx}",
                "props": [
                    ("background-color", color),
                    ("color", text_color),
                    ("font-weight", "600"),
                    ("border", "1px solid #d1d5db"),
                ],
            }
        )
    styler = styler.set_table_styles(base_styles + header_styles)
    styled_html = styler.to_html()
    st.markdown(styled_html, unsafe_allow_html=True)

    st.subheader("Model Overview")
    st.altair_chart(model_net_pnl_chart(summary_df), use_container_width=True)

    st.subheader("Equity & Risk Overview")
    st.altair_chart(multi_model_equity_chart(dfs), use_container_width=True)
    drawdown_model = st.selectbox(
        "Select model for drawdown view",
        options=selected_models,
        format_func=lambda slug: MODEL_LABELS.get(slug, slug),
    )
    drawdown_df = dfs.get(drawdown_model, pd.DataFrame())
    if drawdown_df.empty:
        st.warning("No trade data available for the selected model.")
    else:
        st.altair_chart(
            drawdown_chart(drawdown_df, MODEL_LABELS.get(drawdown_model, drawdown_model)),
            use_container_width=True,
        )

    st.subheader("Model Characteristics")
    st.altair_chart(multi_model_holding_chart(dfs), use_container_width=True)
    st.altair_chart(model_symbol_mix_chart(dfs), use_container_width=True)

    viz_model = st.selectbox(
        "Choose a focus model",
        options=selected_models,
        format_func=lambda slug: MODEL_LABELS.get(slug, slug),
        key="focus_model",
    )
    viz_df = dfs.get(viz_model, pd.DataFrame())
    if viz_df.empty:
        st.warning("No trade data available for the selected model.")
        return

    st.subheader("Trade-Level PnL")
    st.altair_chart(trade_pnl_chart(viz_df), use_container_width=True)
    st.altair_chart(trade_order_count_chart(viz_df), use_container_width=True)
    st.altair_chart(pnl_fee_scatter_chart(viz_df), use_container_width=True)

    st.subheader("Outcome Drivers")
    col_a, col_b = st.columns(2)
    with col_a:
        st.altair_chart(fee_distribution_chart(viz_df), use_container_width=True)
    with col_b:
        st.altair_chart(holding_time_chart(viz_df), use_container_width=True)

    st.subheader("Market Context")
    col_c, col_d = st.columns(2)
    with col_c:
        st.altair_chart(session_heatmap_chart(viz_df), use_container_width=True)
    with col_d:
        st.altair_chart(symbol_contribution_chart(viz_df), use_container_width=True)

    trade_ids = sorted(
        (tid for tid in viz_df["trade_id"].unique() if isinstance(tid, str)),
        key=lambda x: x,
    )
    if not trade_ids:
        return

    st.subheader("Trade Drill-Down")
    selected_trade = st.selectbox("Inspect specific trade", options=trade_ids)

    summary = trade_summary(viz_df, selected_trade)
    if summary:
        cols = st.columns(len(summary))
        for col, (label, value) in zip(cols, summary.items()):
            if isinstance(value, float):
                col.metric(label, f"{value:,.2f}")
            else:
                col.metric(label, value)

    st.altair_chart(trade_timeline_chart(viz_df, selected_trade), use_container_width=True)
    st.altair_chart(trade_order_pnl_chart(viz_df, selected_trade), use_container_width=True)

    detail_df = trade_detail_table(viz_df, selected_trade)
    if not detail_df.empty:
        st.dataframe(
            detail_df.style.format(
                {
                    "Start Position": "{:,.4f}",
                    "Position Size": "{:,.4f}",
                    "Price": "{:,.2f}",
                    "Closed PnL": "{:,.2f}",
                    "Fee": "{:,.2f}",
                    "Net PnL": "{:,.2f}",
                }
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
