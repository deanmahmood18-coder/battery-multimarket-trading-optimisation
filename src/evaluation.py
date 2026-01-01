import numpy as np
import yaml

from src.optimisation_da_only import solve_da_only
from src.optimisation_two_stage import solve_two_stage


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    run = load_yaml("configs/run.yaml")
    mkt = load_yaml("configs/markets.yaml")
    battery = load_yaml("configs/battery.yaml")

    dt = float(run["dt_hours"])
    T = int(run["T"])
    S = int(run["S"])
    seed = int(run["seed"])

    from src.data_loader import load_prices_30min, bootstrap_rt_scenarios

    p_da, p_rt_hist = load_prices_30min()
    p_rt = bootstrap_rt_scenarios(p_da=p_da, p_rt_hist=p_rt_hist, S=S, seed=seed, block_len=48)

    da_res = solve_da_only(p_da=p_da, dt=dt, battery=battery)
    ts_res = solve_two_stage(p_da=p_da, p_rt=p_rt, dt=dt, battery=battery)

    da_pnl = float(da_res["objective"])
    ts_pnl_mean = float(ts_res["scenario_pnl"].mean())
    option_value = ts_pnl_mean - da_pnl

    pnl = ts_res["scenario_pnl"]

    print("DA-only P&L:", round(da_pnl, 2))
    print("Two-stage expected P&L:", round(ts_pnl_mean, 2))
    print("Option value (two-stage - DA-only):", round(option_value, 2))
    print("Two-stage P&L percentiles (p5/p50/p95):",
          round(np.percentile(pnl, 5), 2),
          round(np.percentile(pnl, 50), 2),
          round(np.percentile(pnl, 95), 2))


if __name__ == "__main__":
    main()
