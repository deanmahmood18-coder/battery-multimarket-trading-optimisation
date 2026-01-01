import os
import pandas as pd
import numpy as np


LONDON_TZ = "Europe/London"


def settlement_to_utc(df: pd.DataFrame, date_col: str, period_col: str) -> pd.Series:
    """
    Convert (settlement_date, settlement_period) -> datetime_utc.

    Elexon settlement periods are based on UK local clock time.
    We build a local timestamp and convert to UTC.

    Notes:
    - Most days have 48 periods.
    - Clock-change days can have 46 or 50. We handle localisation with pandas.
    """
    # Build naive local datetime: date + (period-1)*30min
    dt_local_naive = (
        pd.to_datetime(df[date_col], format="%Y-%m-%d")
        + pd.to_timedelta((df[period_col].astype(int) - 1) * 30, unit="min")
    )

    # Localise to Europe/London, handling DST quirks
    # - nonexistent times (spring forward): shift forward
    # - ambiguous times (fall back): infer where possible, else NaT
    dt_local = dt_local_naive.dt.tz_localize(
        LONDON_TZ, nonexistent="shift_forward", ambiguous="infer"
    )

    return dt_local.dt.tz_convert("UTC")


def normalise_da(da_path: str, da_tz: str = "UTC") -> pd.DataFrame:
    """
    Load DA file and output columns:
      datetime_utc, da_price_gbp_mwh
    """
    da = pd.read_csv(da_path)

    if {"settlement_date", "settlement_period", "da_price_gbp_mwh"}.issubset(da.columns):
        da["datetime_utc"] = settlement_to_utc(da, "settlement_date", "settlement_period")
        out = da[["datetime_utc", "da_price_gbp_mwh"]].copy()

    elif {"datetime", "da_price_gbp_mwh"}.issubset(da.columns):
        dt = pd.to_datetime(da["datetime"], errors="raise")

        # If datetime is naive, assume user-provided timezone
        if dt.dt.tz is None:
            if da_tz.upper() == "UTC":
                dt = dt.dt.tz_localize("UTC")
            else:
                dt = dt.dt.tz_localize(LONDON_TZ).dt.tz_convert("UTC")
        else:
            dt = dt.dt.tz_convert("UTC")

        out = pd.DataFrame({"datetime_utc": dt, "da_price_gbp_mwh": da["da_price_gbp_mwh"]})

    else:
        raise ValueError(
            "DA file must contain either:\n"
            "  (datetime, da_price_gbp_mwh)\n"
            "or\n"
            "  (settlement_date, settlement_period, da_price_gbp_mwh)\n"
            f"Found columns: {list(da.columns)}"
        )

    out["da_price_gbp_mwh"] = pd.to_numeric(out["da_price_gbp_mwh"], errors="coerce")
    out = out.dropna(subset=["datetime_utc", "da_price_gbp_mwh"]).sort_values("datetime_utc")
    return out.reset_index(drop=True)


def normalise_sbp(sbp_path: str) -> pd.DataFrame:
    """
    Load SBP file and output columns:
      datetime_utc, rt_price_gbp_mwh
    """
    sbp = pd.read_csv(sbp_path)

    required = {"settlement_date", "settlement_period", "sbp_gbp_mwh"}
    if not required.issubset(sbp.columns):
        raise ValueError(f"SBP file must contain {required}. Found: {list(sbp.columns)}")

    sbp["datetime_utc"] = settlement_to_utc(sbp, "settlement_date", "settlement_period")
    out = sbp[["datetime_utc", "sbp_gbp_mwh"]].copy()
    out = out.rename(columns={"sbp_gbp_mwh": "rt_price_gbp_mwh"})
    out["rt_price_gbp_mwh"] = pd.to_numeric(out["rt_price_gbp_mwh"], errors="coerce")
    out = out.dropna(subset=["datetime_utc", "rt_price_gbp_mwh"]).sort_values("datetime_utc")
    return out.reset_index(drop=True)


def ensure_30min_grid(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Ensure a continuous 30-minute UTC grid.
    If input is hourly, we upsample to 30-min with forward fill.
    """
    df = df.copy()
    df = df.set_index("datetime_utc").sort_index()

    # Force UTC tz-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Build full 30-min range
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="30min", tz="UTC")
    df = df.reindex(full_idx)

    # Fill gaps (hourly DA is common)
    df[value_col] = df[value_col].ffill()

    df = df.reset_index().rename(columns={"index": "datetime_utc"})
    return df


def main():
    os.makedirs("data/processed", exist_ok=True)

    sbp_path = "data/raw/sbp.csv"
    da_path = "data/raw/da.csv"
    out_path = "data/processed/prices_30min.csv"

    if not os.path.exists(sbp_path):
        raise FileNotFoundError(f"Missing {sbp_path}. Run fetch_sbp first.")
    if not os.path.exists(da_path):
        raise FileNotFoundError(f"Missing {da_path}. Create or export DA data to data/raw/da.csv.")

    sbp = normalise_sbp(sbp_path)
    da = normalise_da(da_path, da_tz="UTC")  # change to "LONDON" if your DA datetime column is local time

    # Ensure both are on a consistent 30-min UTC grid
    sbp = ensure_30min_grid(sbp, "rt_price_gbp_mwh")
    da = ensure_30min_grid(da, "da_price_gbp_mwh")

    merged = pd.merge(da, sbp, on="datetime_utc", how="inner")

    # Basic sanity checks
    merged = merged.dropna(subset=["da_price_gbp_mwh", "rt_price_gbp_mwh"])
    merged = merged.sort_values("datetime_utc").reset_index(drop=True)

    merged.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print("Rows:", len(merged))
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
