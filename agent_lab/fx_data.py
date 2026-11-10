"""
fx_data.py - host-side, READ-ONLY market data from the local MetaTrader 5 terminal.

The MetaTrader5 package is Windows-only, so it runs here on the host, never in
the Linux sandboxes. Agents call the `fetch_fx` tool; the host writes a CSV into
that agent's workspace. Only rate/tick COPY functions are used: no orders, no
account details. The MT5 API is not thread-safe, so calls are serialised.

Note: MT5 timestamps are in the BROKER SERVER timezone, not UTC.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_lock = threading.Lock()
_initialized = False

TIMEFRAMES = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
              "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1"}
MAX_BARS = 500_000


def _mt5():
    global _initialized
    import MetaTrader5 as mt5  # noqa: PLC0415  (Windows-only, imported lazily)

    if not _initialized:
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}. Is the MetaTrader 5 terminal running?")
        _initialized = True
    return mt5


def resolve_symbol(mt5, symbol: str) -> str:
    """Brokers decorate names (EURUSD+, EURUSD.m). Accept the plain name."""
    s = symbol.upper().replace("/", "")
    if mt5.symbol_info(s) is not None:
        return s
    matches = [x.name for x in mt5.symbols_get(f"{s}*")]
    if not matches:
        raise ValueError(f"Unknown symbol {symbol!r}")
    return min(matches, key=len)


def fetch_rates(symbol: str, timeframe: str, start: str, end: str | None, out_dir: Path) -> dict:
    tf = timeframe.upper()
    if tf not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {list(TIMEFRAMES)}")
    t0 = datetime.fromisoformat(start)
    t1 = datetime.fromisoformat(end) if end else datetime.now() + timedelta(days=1)
    with _lock:
        mt5 = _mt5()
        name = resolve_symbol(mt5, symbol)
        mt5.symbol_select(name, True)
        rates = mt5.copy_rates_range(name, getattr(mt5, TIMEFRAMES[tf]), t0, t1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data for {name} {tf} {start}..{end}: {mt5.last_error()}")
        info = mt5.symbol_info(name)
    df = pd.DataFrame(rates[:MAX_BARS])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})[["time", "open", "high", "low", "close", "volume", "spread"]]
    # Agents misread 'spread' (points) as pips; give it in price units too, so a cost is just `close - spread_price`.
    df["spread_price"] = (df["spread"] * info.point).round(info.digits + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol.upper()}_{tf}_{t0:%Y%m%d}_{df['time'].iloc[-1]:%Y%m%d}.csv"
    df.to_csv(path, index=False)
    return {
        "file": f"data/{path.name}",
        "symbol": name, "timeframe": tf, "rows": len(df),
        "first": str(df["time"].iloc[0]), "last": str(df["time"].iloc[-1]),
        "point": info.point, "digits": info.digits,
        "columns": list(df.columns),
        "pip": info.point * 10 if info.digits in (3, 5) else info.point,
        "note": "time = BROKER SERVER TIME, not UTC (don't label hours as UTC). spread = points; "
                "spread_price = spread in price units (subtract it as the round-trip cost). 1 pip = 'pip' price units.",
    }


if __name__ == "__main__":
    import json
    import tempfile

    print(json.dumps(fetch_rates("EURUSD", "H1", "2026-10-01", None, Path(tempfile.mkdtemp())), indent=2))
