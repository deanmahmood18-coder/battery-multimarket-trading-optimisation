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

    from src.data_loader import load_prices_30min, bootstrap_rt_scenarios

    p_da, p_rt_hist = load_prices_30min()
    p_rt = bootstrap_rt_scenarios(p_da, p_rt_hist, S=S, seed=seed, block_len=48)

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

    # Price scenarios: DA vs a few RT paths
    plt.figure(figsize=(10, 4))
    plt.plot(p_da, label="DA", linewidth=2)
    for i in range(min(5, p_rt.shape[0])):
        plt.plot(p_rt[i], alpha=0.35, linewidth=1)
    plt.title("DA price path with RT scenario samples")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    outpath = "outputs/charts/price_scenarios.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")

    # Day-ahead schedule (two-stage non-anticipative dispatch)
    ch = res["ch_DA"]
    dis = res["dis_DA"]
    plt.figure(figsize=(10, 4))
    plt.plot(ch, label="Charge (MW)", color="#3a6ea5")
    plt.plot(dis, label="Discharge (MW)", color="#f5a623")
    plt.title("DA schedule (two-stage)")
    plt.xlabel("Time step")
    plt.ylabel("Power (MW)")
    plt.legend()
    outpath = "outputs/charts/da_schedule.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
