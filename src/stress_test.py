import argparse
import os
import yaml
import numpy as np
import pandas as pd

from src.optimisation_da_only import solve_da_only
from src.optimisation_two_stage import solve_two_stage
from src.scenario_generation import generate_da_prices, generate_rt_scenarios


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_regime(p_da, S, dt, seed, battery, da_vol, rt_noise_vol, spike_prob, spike_size, da_res):
    # RT scenarios
    p_rt = generate_rt_scenarios(
        p_da=p_da,
        S=S,
        noise_vol=rt_noise_vol,
        spike_prob=spike_prob,
        spike_size=spike_size,
        seed=seed + 1
    )

    ts_res = solve_two_stage(p_da=p_da, p_rt=p_rt, dt=dt, battery=battery)

    da_pnl = float(da_res["objective"])
    pnl = ts_res["scenario_pnl"]
    ts_mean = float(pnl.mean())
    option_value = ts_mean - da_pnl

    return {
        "da_vol": da_vol,
        "rt_noise_vol": rt_noise_vol,
        "spike_prob": spike_prob,
        "spike_size": spike_size,
        "DA_PnL": da_pnl,
        "TS_mean_PnL": ts_mean,
        "OptionValue": option_value,
        "TS_p05": float(np.percentile(pnl, 5)),
        "TS_p50": float(np.percentile(pnl, 50)),
        "TS_p95": float(np.percentile(pnl, 95)),
        "TS_worst": float(pnl.min()),
        "TS_best": float(pnl.max()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-regime stress test.")
    parser.add_argument("--fast", action="store_true", help="Use smaller grids and fewer scenarios.")
    parser.add_argument("--S", type=int, default=None, help="Override number of RT scenarios.")
    parser.add_argument("--T", type=int, default=None, help="Override number of time steps.")
    return parser.parse_args()


def main():
    os.makedirs("outputs/tables", exist_ok=True)

    run = load_yaml("configs/run.yaml")
    mkt = load_yaml("configs/markets.yaml")
    battery = load_yaml("configs/battery.yaml")

    args = parse_args()

    T = int(run["T"])
    S = int(run["S"])
    dt = float(run["dt_hours"])
    seed = int(run["seed"])

    da_base = float(mkt["da_base_price"])
    spike_size = float(mkt["spike_size"])

    # Regime grid (edit these freely)
    da_vol_grid = [10, 15, 25]
    rt_noise_grid = [5, 10, 15]
    spike_prob_grid = [0.00, 0.01, 0.03, 0.05]

    if args.fast:
        # Smaller grid and scenario count for quick iterations.
        da_vol_grid = [10, 20]
        rt_noise_grid = [5, 10]
        spike_prob_grid = [0.00, 0.03]
        S = min(S, 30)
        T = min(T, 96)

    if args.S is not None:
        S = int(args.S)
    if args.T is not None:
        T = int(args.T)

    print(f"Stress test config: T={T}, S={S}, da_vol_grid={da_vol_grid}, rt_noise_grid={rt_noise_grid}, spike_prob_grid={spike_prob_grid}")

    rows = []
    for da_vol in da_vol_grid:
        # DA path and DA-only solve are reused across RT regimes for the same DA volatility.
        p_da = generate_da_prices(T=T, base=da_base, vol=da_vol, seed=seed)
        da_res = solve_da_only(p_da=p_da, dt=dt, battery=battery)
        for rt_noise_vol in rt_noise_grid:
            for spike_prob in spike_prob_grid:
                rows.append(
                    run_regime(
                        p_da=p_da,
                        S=S,
                        dt=dt,
                        seed=seed,
                        battery=battery,
                        da_vol=da_vol,
                        rt_noise_vol=rt_noise_vol,
                        spike_prob=spike_prob,
                        spike_size=spike_size,
                        da_res=da_res,
                    )
                )
                print(f"Done regime: da_vol={da_vol}, rt_noise={rt_noise_vol}, spike_prob={spike_prob}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["spike_prob", "rt_noise_vol", "da_vol"]).reset_index(drop=True)

    out_csv = "outputs/tables/stress_test_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Print top 10 by option value (quick insight)
    top = df.sort_values("OptionValue", ascending=False).head(10)
    print("\nTop 10 regimes by OptionValue:")
    print(top[["da_vol", "rt_noise_vol", "spike_prob", "OptionValue", "TS_p05", "TS_p50", "TS_p95"]].to_string(index=False))


if __name__ == "__main__":
    main()
