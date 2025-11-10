"""Group Hyperliquid fills into full-position trade pictures.

Rows from the Hyperliquid fills CSV are grouped so that every open/close leg
belonging to a continuous position (per coin & direction) shares a unique
trade identifier. A new `trade_id` column is appended while leaving all
original columns untouched.

Usage:
    python3 hyperliquid_trade_picture.py \
        --input hyperliquid_trades.csv \
        --output hyperliquid_trade_picture.csv \
        --show-summary
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INPUT = "hyperliquid_trades.csv"
DEFAULT_OUTPUT = "hyperliquid_trade_picture.csv"
MODEL_FILES = {
    "qwen3": ("hyperliquid_trades_qwen3.csv", "hyperliquid_trade_picture_qwen3.csv"),
    "deepseek": ("hyperliquid_trades_deepseek.csv", "hyperliquid_trade_picture_deepseek.csv"),
    "gemini": ("hyperliquid_trades_gemini.csv", "hyperliquid_trade_picture_gemini.csv"),
    "claude": ("hyperliquid_trades_claude.csv", "hyperliquid_trade_picture_claude.csv"),
    "grok4": ("hyperliquid_trades_grok4.csv", "hyperliquid_trade_picture_grok4.csv"),
    "gpt5": ("hyperliquid_trades_gpt5.csv", "hyperliquid_trade_picture_gpt5.csv"),
}
TOLERANCE = 1e-8
POSITION_DRIFT_TOLERANCE = 5e-3
POSITION_RELATIVE_TOLERANCE = 0.15


@dataclass
class Fill:
    index: int
    row: Dict[str, str]
    coin: str
    side: str  # "long" or "short"
    action: str  # "open" or "close"
    size: float
    start_position: float
    time_ms: int
    hash: str


@dataclass
class PositionState:
    position: float = 0.0
    current_group: Optional[int] = None
    next_group: int = 1


def parse_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_int(value: Any) -> int:
    try:
        return int(parse_float(value))
    except (TypeError, ValueError):
        return 0


def interpret_dir(raw_dir: Any) -> Tuple[Optional[str], Optional[str]]:
    if raw_dir is None:
        return None, None
    text = str(raw_dir).strip().lower()
    if text.startswith("open"):
        action = "open"
    elif text.startswith("close"):
        action = "close"
    else:
        return None, None

    if "long" in text:
        side = "long"
    elif "short" in text:
        side = "short"
    else:
        side = None

    return action, side


def almost_equal(a: float, b: float, tol: float = TOLERANCE) -> bool:
    return abs(a - b) <= max(tol, (abs(a) + abs(b)) * 1e-9)


def zero_if_close(value: float) -> float:
    return 0.0 if abs(value) <= TOLERANCE else value


def ms_to_readable(value: str) -> str:
    if not value:
        return ""
    try:
        millis = float(value)
    except (TypeError, ValueError):
        return value
    seconds = millis / 1000.0
    if seconds <= 0:
        return value
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"


def load_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for idx, row in enumerate(reader):
            row["_index"] = str(idx)
            rows.append(row)
    return fieldnames, rows


def prepare_fills(rows: List[Dict[str, str]]) -> List[Fill]:
    fills: List[Fill] = []
    for row in rows:
        action, side = interpret_dir(row.get("dir"))
        if action is None or side is None:
            continue
        size = parse_float(row.get("sz"))
        if size <= 0:
            continue
        start_pos = parse_float(row.get("startPosition"))
        time_ms = parse_int(row.get("time"))
        fills.append(
            Fill(
                index=int(row["_index"]),
                row=row,
                coin=str(row.get("coin", "")),
                side=side,
                action=action,
                size=size,
                start_position=start_pos,
                time_ms=time_ms,
                hash=str(row.get("hash", "")),
            )
        )
    fills.sort(key=fill_sort_key)
    return fills


def fill_sort_key(fill: Fill) -> Tuple[int, int, float, str]:
    primary = fill.time_ms
    if fill.action == "open":
        order = 0
        secondary = fill.start_position if fill.side == "long" else -fill.start_position
    else:
        order = 1
        secondary = -fill.start_position if fill.side == "long" else fill.start_position
    return (primary, order, secondary, fill.hash)


def assign_trade_ids(fills: List[Fill]) -> Dict[int, str]:
    states: Dict[Tuple[str, str], PositionState] = {}
    assignments: Dict[int, str] = {}

    for fill in fills:
        key = (fill.coin, fill.side)
        state = states.setdefault(key, PositionState())
        expected = state.position
        actual = fill.start_position
        if not almost_equal(expected, actual):
            drift = abs(expected - actual)
            max_abs = max(abs(expected), abs(actual), 1.0)
            tolerance = max(POSITION_DRIFT_TOLERANCE, POSITION_RELATIVE_TOLERANCE * max_abs)
            if drift <= tolerance:
                state.position = actual
                expected = actual
            else:
                print(
                    f"[trade_picture] realigning {fill.coin} {fill.side} state: expected {expected}, "
                    f"saw {actual} (idx {fill.index})"
                )
                state.position = actual
                expected = actual

        if fill.action == "open":
            if state.current_group is None or almost_equal(expected, 0.0):
                state.current_group = state.next_group
                state.next_group += 1
            trade_id = f"{fill.coin}_{fill.side}_{state.current_group}"
            if fill.side == "long":
                state.position = expected + fill.size
            else:
                state.position = expected - fill.size
        else:  # close
            if state.current_group is None:
                state.current_group = state.next_group
                state.next_group += 1
            trade_id = f"{fill.coin}_{fill.side}_{state.current_group}"
            if fill.side == "long":
                state.position = expected - fill.size
            else:
                state.position = expected + fill.size
            state.position = zero_if_close(state.position)
            if almost_equal(state.position, 0.0):
                state.current_group = None

        state.position = zero_if_close(state.position)
        assignments[fill.index] = trade_id

    return assignments


def write_grouped_rows(
    fieldnames: List[str],
    rows: List[Dict[str, str]],
    trade_ids: Dict[int, str],
    output_path: str,
) -> None:
    base_fields = [name for name in fieldnames if name]

    if "Position size" not in base_fields:
        insert_after = base_fields.index("dir") + 1 if "dir" in base_fields else len(base_fields)
        base_fields.insert(insert_after, "Position size")

    if "trade_id" in base_fields:
        base_fields.remove("trade_id")
    base_fields.insert(1 if base_fields else 0, "trade_id")

    if "fee" in base_fields:
        base_fields.remove("fee")
        if "closedPnl" in base_fields:
            insert_at = base_fields.index("closedPnl") + 1
        else:
            insert_at = base_fields.index("dir") + 1 if "dir" in base_fields else len(base_fields)
        base_fields.insert(insert_at, "fee")

    def sort_key(row: Dict[str, str]) -> Tuple[str, float, int]:
        tid = trade_ids.get(int(row["_index"]), "")
        time_str = row.get("time", "")
        try:
            time_val = float(time_str)
        except (TypeError, ValueError):
            time_val = float("inf")
        return tid, time_val, int(row["_index"])

    sorted_rows = sorted(rows, key=sort_key)

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields)
        writer.writeheader()
        for row in sorted_rows:
            idx = int(row["_index"])
            trade_id = trade_ids.get(idx, "")
            out_row = {}
            for name in base_fields:
                if name == "trade_id":
                    out_row[name] = trade_id
                elif name == "time":
                    out_row[name] = ms_to_readable(row.get(name, ""))
                elif name == "Position size":
                    direction = str(row.get("dir", "")).lower()
                    start_pos = parse_float(row.get("startPosition"))
                    size = parse_float(row.get("sz"))
                    if "open long" in direction:
                        out_row[name] = start_pos + size
                    elif "close long" in direction:
                        out_row[name] = start_pos - size
                    elif "open short" in direction:
                        out_row[name] = start_pos - size
                    elif "close short" in direction:
                        out_row[name] = start_pos + size
                    else:
                        out_row[name] = ""
                else:
                    out_row[name] = row.get(name, "")
            writer.writerow(out_row)


def process_trade_picture(input_path: str, output_path: str, show_summary: bool = False, label: Optional[str] = None) -> None:
    try:
        fieldnames, rows = load_rows(input_path)
    except FileNotFoundError:
        print(f"Skipping {label or input_path}: file not found")
        return

    fills = prepare_fills(rows)
    trade_ids = assign_trade_ids(fills)
    write_grouped_rows(fieldnames, rows, trade_ids, output_path)

    if show_summary:
        total = len(fills)
        grouped = len(set(trade_ids.values()))
        tag = label or output_path
        print(f"{tag}: fills processed = {total}, trade groups = {grouped}")
        unmatched = sum(1 for row in rows if int(row["_index"]) not in trade_ids)
        if unmatched:
            print(f"{tag}: warning - {unmatched} rows were skipped (missing dir or size)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group Hyperliquid fills into full trade pictures.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input fills CSV (default: hyperliquid_trades.csv)")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV with trade_id column (default: hyperliquid_trade_picture.csv)",
    )
    parser.add_argument("--show-summary", action="store_true", help="Print summary statistics")
    parser.add_argument("--all-models", action="store_true", help="Process all predefined model CSVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_models:
        for slug, (input_path, output_path) in MODEL_FILES.items():
            process_trade_picture(input_path, output_path, show_summary=args.show_summary, label=slug)
        return

    process_trade_picture(args.input, args.output, show_summary=args.show_summary)


if __name__ == "__main__":
    main()
