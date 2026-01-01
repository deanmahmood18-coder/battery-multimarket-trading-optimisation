import os
import yaml
import numpy as np
import pandas as pd

from src.scenario_generation import generate_da_prices, generate_rt_scenarios


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    os.makedirs("outputs/tables", exist_ok=True)

    run = load_yaml("configs/run.yaml")
    mkt = load_yaml("configs/markets.yaml")

    T = int(run["T"])
    S = int(run["S"])
    seed = int(run["seed"])

    p_da = generate_da_prices(
        T=T,
        base=float(mkt["da_base_price"]),
        vol=float(mkt["da_vol"]),
        seed=seed,
    )
    p_rt = generate_rt_scenarios(
        p_da=p_da,
        S=S,
        noise_vol=float(mkt["rt_noise_vol"]),
        spike_prob=float(mkt["spike_prob"]),
        spike_size=float(mkt["spike_size"]),
        seed=seed + 1,
    )

    spreads = p_rt - p_da[None, :]

    # Spike proxy: count “extreme” spread moves
    # (Because we synthetically inject spikes, this is a good sanity check)
    extreme = np.abs(spreads) > float(mkt["spike_size"]) * 0.5
    extreme_rate = extreme.mean()

    summary = {
        "T": T,
        "S": S,
        "DA_mean": float(p_da.mean()),
        "DA_std": float(p_da.std()),
        "RT_mean": float(p_rt.mean()),
        "RT_std": float(p_rt.std()),
        "Spread_mean": float(spreads.mean()),
        "Spread_std": float(spreads.std()),
        "Spread_p01": float(np.percentile(spreads, 1)),
        "Spread_p99": float(np.percentile(spreads, 99)),
        "Extreme_spread_rate": float(extreme_rate),
    }

    df = pd.DataFrame([summary])
    out = "outputs/tables/scenario_diagnostics.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
