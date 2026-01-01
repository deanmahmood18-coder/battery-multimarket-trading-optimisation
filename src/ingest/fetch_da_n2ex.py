import argparse
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


BASE = "https://data-api.nordpoolgroup.com"


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def fetch_price_curves_uk(delivery_date: str) -> Dict[str, Any]:
    """
    Nord Pool v2 data API lists an endpoint for N2EX Day-Ahead PriceCurves for UK. :contentReference[oaicite:2]{index=2}
    We call it for a single delivery date and then normalise.

    Note: If you get 401/403, your account/subscription may be required. In that case use Route B (export CSV).
    """
    url = f"{BASE}/api/v2/Auction/N2EX_DayAhead/PriceCurves/UK"
    r = requests.get(url, params={"deliveryDate": delivery_date}, timeout=60)
    r.raise_for_status()
    return r.json()


def normalise_to_hourly(df_json: Dict[str, Any], delivery_date: str) -> pd.DataFrame:
    """
    Tries to extract an hourly price series from typical Nord Pool response shapes.

    Because response schemas can vary, we:
    - search for a list-like structure of points
    - expect each point to contain a timestamp or hour + price
    - fall back with a helpful error printing top-level keys
    """
    # Common patterns: {"data": {...}} or direct object
    obj = df_json.get("data", df_json)

    # Try a few likely locations for curve/points arrays
    candidates = [
        obj.get("priceCurves"),
        obj.get("curves"),
        obj.get("priceCurve"),
        obj.get("rows"),
        obj.get("points"),
    ]

    points = None
    for c in candidates:
        if isinstance(c, list) and len(c) > 0:
            points = c
            break

    if points is None:
        # Sometimes it's nested one level deeper
        for k, v in obj.items() if isinstance(obj, dict) else []:
            if isinstance(v, list) and v and isinstance(v[0], dict) and ("price" in v[0] or "value" in v[0]):
                points = v
                break

    if points is None:
        raise ValueError(
            "Could not find price points in Nord Pool response.\n"
            f"Top-level keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}\n"
            "Use Route B (export from Nord Pool page) if this endpoint requires subscription."
        )

    rows = []
    for p in points:
        if not isinstance(p, dict):
            continue
        # Try a few likely field names
        ts = p.get("time") or p.get("timestamp") or p.get("startTime") or p.get("deliveryStart")
        price = p.get("price") or p.get("value") or p.get("priceEur") or p.get("priceGbp")
        hour = p.get("hour")

        rows.append({"delivery_date": delivery_date, "timestamp": ts, "hour": hour, "price": price})

    df = pd.DataFrame(rows)
    if df["price"].isna().all():
        raise ValueError(f"Found points but could not locate price field. Example point keys: {list(points[0].keys())}")

    # Build datetime (prefer timestamp if present)
    if df["timestamp"].notna().any():
        dt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.assign(datetime_utc=dt)
    else:
        # Fallback: use delivery_date + hour
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df["datetime_utc"] = pd.to_datetime(df["delivery_date"]) + pd.to_timedelta(df["hour"], unit="h")
        df["datetime_utc"] = df["datetime_utc"].dt.tz_localize("UTC")

    df["da_price_gbp_mwh"] = pd.to_numeric(df["price"], errors="coerce")

    out = df[["datetime_utc", "da_price_gbp_mwh"]].dropna().sort_values("datetime_utc").reset_index(drop=True)
    return out


def expand_hourly_to_settlement_periods(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Your merge pipeline expects 30-min periods. If DA is hourly, we duplicate each hour to two half-hours.
    """
    df = df_hourly.copy()
    df = df.set_index("datetime_utc").sort_index()
    # upsample to 30-min, forward-fill
    full_idx = pd.date_range(df.index.min(), df.index.max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=30), freq="30min", tz="UTC")
    df = df.reindex(full_idx)
    df["da_price_gbp_mwh"] = df["da_price_gbp_mwh"].ffill()
    df = df.reset_index().rename(columns={"index": "datetime_utc"})
    return df


def write_da_csv(df_30m: pd.DataFrame, out_path: str):
    # Convert UTC datetime to settlement_date/period in Europe/London clock, since SBP uses settlement periods
    dt_local = df_30m["datetime_utc"].dt.tz_convert("Europe/London")
    settlement_date = dt_local.dt.date.astype(str)

    # period = 1 + minutes since midnight / 30
    minutes = dt_local.dt.hour * 60 + dt_local.dt.minute
    settlement_period = (minutes // 30 + 1).astype(int)

    out = pd.DataFrame(
        {
            "settlement_date": settlement_date,
            "settlement_period": settlement_period,
            "da_price_gbp_mwh": df_30m["da_price_gbp_mwh"].astype(float),
        }
    ).sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(out.head(10).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default="data/raw/da.csv")
    args = p.parse_args()

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)

    hourly_parts = []
    for d in daterange(d0, d1):
        ds = d.isoformat()
        payload = fetch_price_curves_uk(ds)
        df_hourly = normalise_to_hourly(payload, ds)
        hourly_parts.append(df_hourly)
        print(f"Fetched DA for {ds}: {len(df_hourly)} rows")

    df_hourly_all = pd.concat(hourly_parts, ignore_index=True).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc")
    df_30m = expand_hourly_to_settlement_periods(df_hourly_all)

    # Ensure tz-aware
    if df_30m["datetime_utc"].dt.tz is None:
        df_30m["datetime_utc"] = df_30m["datetime_utc"].dt.tz_localize("UTC")

    write_da_csv(df_30m, args.out)


if __name__ == "__main__":
    main()
