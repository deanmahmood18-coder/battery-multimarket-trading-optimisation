# src/ingest/fetch_sbp.py

import argparse
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def pick(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def fetch_system_prices_for_date(settlement_date: str) -> List[Dict[str, Any]]:
    """
    Fetch GB System Prices (SBP/SSP) for a given settlement date.

    Endpoint (Elexon BMRS/Insights v1 style):
      /balancing/settlement/system-prices/{settlementDate}

    Returns a list of rows (typically 48 rows for the day).
    """
    url = f"{BASE_URL}/balancing/settlement/system-prices/{settlement_date}"
    r = requests.get(url, params={"format": "json"}, timeout=60)
    r.raise_for_status()
    payload = r.json()

    # Common shape: {"data": [...], "metadata": {...}}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]

    # Sometimes APIs return a list directly
    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unexpected response shape: {type(payload)} from {url}")


def normalise_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalise API rows into:
      settlement_date, settlement_period, sbp_gbp_mwh, ssp_gbp_mwh (if present)

    If field names differ, the script prints the first-row keys to help you map.
    """
    out = []
    for r in rows:
        settlement_date = pick(r, ["settlementDate", "settlement_date", "date"])
        period = pick(r, ["settlementPeriod", "settlement_period", "period", "settlementperiod"])

        # Try common field names for SBP/SSP
        sbp = pick(r, ["systemBuyPrice", "SystemBuyPrice", "sbp", "SBP", "system_buy_price"])
        ssp = pick(r, ["systemSellPrice", "SystemSellPrice", "ssp", "SSP", "system_sell_price"])

        out.append(
            {
                "settlement_date": settlement_date,
                "settlement_period": period,
                "sbp_gbp_mwh": sbp,
                "ssp_gbp_mwh": ssp,
            }
        )

    df = pd.DataFrame(out)

    # Validate we actually captured SBP
    if df["sbp_gbp_mwh"].isna().all():
        first_keys = list(rows[0].keys()) if rows else []
        raise ValueError(
            "Could not locate SBP in API response. "
            f"First row keys were: {first_keys}. "
            "Update the SBP key mapping in normalise_rows()."
        )

    df["settlement_period"] = pd.to_numeric(df["settlement_period"], errors="coerce").astype("Int64")
    df["sbp_gbp_mwh"] = pd.to_numeric(df["sbp_gbp_mwh"], errors="coerce")
    df["ssp_gbp_mwh"] = pd.to_numeric(df["ssp_gbp_mwh"], errors="coerce")

    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default="data/raw/sbp.csv")
    args = p.parse_args()

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)

    all_rows: List[Dict[str, Any]] = []
    for d in daterange(d0, d1):
        ds = d.isoformat()
        rows = fetch_system_prices_for_date(ds)
        all_rows.extend(rows)
        print(f"Fetched {len(rows)} rows for {ds}")

    df = normalise_rows(all_rows)

    # Ensure output folders exist
    out_path = args.out
    pd.options.mode.chained_assignment = None  # just to keep logs clean
    df.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
