"""Utility to fetch Hyperliquid perpetual trade history for a wallet.

Run directly:
    python3 hyperliquid_fetch.py --address 0xc20ac4dc4188660cbf555448af52694ca62b0734

Environment variable overrides:
    HYPERLIQUID_ADDRESS
    HYPERLIQUID_CHAIN (default: ARBITRUM)
    HYPERLIQUID_LIMIT (default: 500)
    HYPERLIQUID_START_TIME / HYPERLIQUID_END_TIME (epoch seconds)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:  # pandas optional; only needed for dataframe output
    pd = None

import requests


DEFAULT_CHAIN = "ARBITRUM"
MODEL_ADDRESSES = {
    "qwen3": "0x7a8fd8bba33e37361ca6b0cb4518a44681bad2f3",
    "deepseek": "0xc20ac4dc4188660cbf555448af52694ca62b0734",
    "gemini": "0x1b7a7d099a670256207a30dd0ae13d35f278010f",
    "claude": "0x59fa085d106541a834017b97060bcbbb0aa82869",
    "grok4": "0x56d652e62998251b56c8398fb11fcfe464c08f84",
    "gpt5": "0x67293d914eafb26878534571add81f6bd2d9fe06",
}
DEFAULT_ADDRESS = MODEL_ADDRESSES["qwen3"]
DEFAULT_CSV_PATH = "hyperliquid_trades.csv"


def _is_zero_hash(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return text != "" and set(text) <= {"0"}


def _is_dust_event(fill: Dict[str, Any]) -> bool:
    direction = str(fill.get("dir", "")).strip().lower()
    if "dust conversion" in direction:
        return True
    hash_val = fill.get("hash") or fill.get("txHash")
    return _is_zero_hash(hash_val)


def fetch_hyperliquid_trades(
    address: str,
    chain: str = DEFAULT_CHAIN,
    limit: Optional[int] = 500,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[dict]:
    """Fetch per-address trade history from the Hyperliquid info API."""
    payload: Dict[str, Any] = {"type": "userFills", "user": address, "chain": chain}
    if start_time is not None:
        payload["startTime"] = start_time
    if end_time is not None:
        payload["endTime"] = end_time
    if limit is not None:
        payload["n"] = limit

    resp = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fills: Any
    if isinstance(data, dict):
        fills = data.get("fills") or data.get("data") or []
    else:
        fills = data

    if not isinstance(fills, list):
        raise ValueError("Unexpected Hyperliquid response payload")

    filtered = [fill for fill in fills if not _is_dust_event(fill)]
    return filtered


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def trades_to_dataframe(trades: List[dict]) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for DataFrame output. Install it with `pip install pandas`.")
    if not trades:
        return pd.DataFrame()
    return pd.json_normalize(trades)


def _stringify(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return value


def write_trades_to_csv(trades: List[dict], path: str) -> None:
    if pd is not None:
        df = trades_to_dataframe(trades)
        df.to_csv(path, index=False)
        return

    if not trades:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return

    fieldnames = sorted({key for trade in trades for key in trade.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            row = {key: _stringify(trade.get(key)) for key in fieldnames}
            writer.writerow(row)


def model_slug(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


def default_csv_path_for_model(slug: str) -> str:
    return f"hyperliquid_trades_{slug}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid trade history for a wallet.")
    parser.add_argument(
        "--address",
        default=os.environ.get("HYPERLIQUID_ADDRESS", DEFAULT_ADDRESS),
        help="Wallet address",
    )
    parser.add_argument("--chain", default=os.environ.get("HYPERLIQUID_CHAIN", DEFAULT_CHAIN), help="Hyperliquid chain name")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("HYPERLIQUID_LIMIT", "500")), help="Max records to fetch")
    parser.add_argument("--start-time", type=float, default=parse_float(os.environ.get("HYPERLIQUID_START_TIME")), help="Start time (epoch seconds)")
    parser.add_argument("--end-time", type=float, default=parse_float(os.environ.get("HYPERLIQUID_END_TIME")), help="End time (epoch seconds)")
    parser.add_argument("--preview", type=int, default=int(os.environ.get("HYPERLIQUID_PREVIEW", "10")), help="Number of rows to display")
    parser.add_argument("--dump-json", action="store_true", help="Print full JSON payload instead of preview")
    parser.add_argument("--show-dataframe", action="store_true", help="Pretty-print the data as a pandas DataFrame")
    parser.add_argument(
        "--to-csv",
        default=os.environ.get("HYPERLIQUID_CSV_PATH"),
        help="Output path to write the fills as CSV (defaults to hyperliquid_trades.csv)",
    )
    parser.add_argument("--all-models", action="store_true", help="Fetch and save CSVs for all predefined model addresses")
    args = parser.parse_args()

    if args.all_models:
        for name, address in MODEL_ADDRESSES.items():
            slug = model_slug(name)
            csv_path = default_csv_path_for_model(slug)
            trades = fetch_hyperliquid_trades(
                address=address,
                chain=args.chain,
                limit=args.limit,
                start_time=args.start_time,
                end_time=args.end_time,
            )
            write_trades_to_csv(trades, csv_path)
            print(f"{name}: fetched {len(trades)} trades -> {csv_path}")
        return

    if not args.address:
        parser.error(
            "Wallet address must be provided via --address, HYPERLIQUID_ADDRESS, "
            f"or the built-in default ({DEFAULT_ADDRESS})."
        )

    trades = fetch_hyperliquid_trades(
        address=args.address,
        chain=args.chain,
        limit=args.limit,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(f"Fetched {len(trades)} Hyperliquid trades for {args.address} on {args.chain}")

    if args.dump_json:
        print(json.dumps(trades, indent=2, ensure_ascii=True))
    else:
        preview = trades[: max(args.preview, 0)]
        for idx, trade in enumerate(preview, start=1):
            print(f"{idx:03d}: {json.dumps(trade, separators=(',', ':'), ensure_ascii=True)}")

    if args.show_dataframe:
        try:
            df = trades_to_dataframe(trades)
        except RuntimeError as exc:
            print(exc)
        else:
            print(df.head(args.preview if args.preview > 0 else None))

    csv_path = args.to_csv or DEFAULT_CSV_PATH
    write_trades_to_csv(trades, csv_path)
    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
