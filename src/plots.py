import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.optimisation_two_stage import solve_two_stage
from src.scenario_generation import generate_da_prices, generate_rt_scenarios


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

    p_da = generate_da_prices(T=T, base=float(mkt["da_base_price"]), vol=float(mkt["da_vol"]), seed=seed)
    p_rt = generate_rt_scenarios(
        p_da=p_da, S=S,
        noise_vol=float(mkt["rt_noise_vol"]),
        spike_prob=float(mkt["spike_prob"]),
        spike_size=float(mkt["spike_size"]),
        seed=seed + 1
    )

    res = solve_two_stage(p_da=p_da, p_rt=p_rt, dt=dt, battery=battery)
    pnl = res["scenario_pnl"]

    os.makedirs("outputs/charts", exist_ok=True)

    plt.figure()
    plt.hist(pnl, bins=25)
    plt.title("Two-stage realised P&L distribution")
    plt.xlabel("P&L")
    plt.ylabel("Frequency")
    outpath = "outputs/charts/pnl_distribution.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
