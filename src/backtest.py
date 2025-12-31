import yaml
import numpy as np
from src.scenario_generation import generate_da_prices, generate_rt_scenarios

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    run = load_yaml("configs/run.yaml")
    mkt = load_yaml("configs/markets.yaml")

    T = int(run["T"])
    S = int(run["S"])
    seed = int(run["seed"])

    p_da = generate_da_prices(
        T=T,
        base=float(mkt["da_base_price"]),
        vol=float(mkt["da_vol"]),
        seed=seed
    )

    p_rt = generate_rt_scenarios(
        p_da=p_da,
        S=S,
        noise_vol=float(mkt["rt_noise_vol"]),
        spike_prob=float(mkt["spike_prob"]),
        spike_size=float(mkt["spike_size"]),
        seed=seed + 1
    )

    print("DA prices shape:", p_da.shape)
    print("RT scenarios shape:", p_rt.shape)
    print("DA sample:", np.round(p_da[:5], 2))
    print("RT sample (scenario 0):", np.round(p_rt[0, :5], 2))

if __name__ == "__main__":
    main()
