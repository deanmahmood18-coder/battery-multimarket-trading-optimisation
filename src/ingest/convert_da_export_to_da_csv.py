import re
import pandas as pd


def pick_col(cols, patterns):
    for pat in patterns:
        r = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if r.search(c):
                return c
    return None


def main():
    inp = "data/raw/da_export.csv"
    out = "data/raw/da.csv"

    df = pd.read_csv(inp)
    cols = list(df.columns)

    # Try common column names seen in price exports
    date_col = pick_col(cols, [r"^date$", r"delivery\s*date", r"day"])
    time_col = pick_col(cols, [r"^time$", r"hour", r"from", r"start"])
    dt_col   = pick_col(cols, [r"datetime", r"timestamp", r"delivery\s*start"])
    price_col = pick_col(cols, [r"price", r"gbp", r"£", r"system\s*price"])

    if price_col is None:
        raise ValueError(f"Could not find a price column in {cols}. Rename the price column to include 'price'.")

    # Build a datetime column (UTC) from either:
    # - a single datetime column, or
    # - date + time/hour columns
    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], errors="raise", utc=True)
    else:
        if date_col is None or time_col is None:
            raise ValueError(
                "Could not construct datetime. Your export must contain either:\n"
                " - a datetime/timestamp column, or\n"
                " - a date column + a time/hour column.\n"
                f"Columns found: {cols}"
            )
        # Combine date + time
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        dt = pd.to_datetime(combined, errors="raise", utc=True)

    out_df = pd.DataFrame(
        {
            "datetime_utc": dt,
            "da_price_gbp_mwh": pd.to_numeric(df[price_col], errors="coerce"),
        }
    ).dropna()

    # Convert datetime_utc -> (settlement_date, settlement_period) in Europe/London clock
    dt_local = out_df["datetime_utc"].dt.tz_convert("Europe/London")
    out_df["settlement_date"] = dt_local.dt.date.astype(str)

    minutes = dt_local.dt.hour * 60 + dt_local.dt.minute
    out_df["settlement_period"] = (minutes // 30 + 1).astype(int)

    final = out_df[["settlement_date", "settlement_period", "da_price_gbp_mwh"]].copy()
    final = final.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    final.to_csv(out, index=False)
    print(f"Wrote: {out}")
    print(final.head(10).to_string(index=False))
    print("\nColumns used from export:")
    print({"datetime_col": dt_col, "date_col": date_col, "time_col": time_col, "price_col": price_col})


if __name__ == "__main__":
    main()
