import numpy as np
import pandas as pd


def load_prices_30min(path: str = "data/processed/prices_30min.csv"):
    """
    Returns:
      p_da: (T,) numpy array
      p_rt_hist: (T,) numpy array (SBP)
    """
    df = pd.read_csv(path)
    if not {"da_price_gbp_mwh", "rt_price_gbp_mwh"}.issubset(df.columns):
        raise ValueError(f"Expected columns da_price_gbp_mwh and rt_price_gbp_mwh in {path}")

    p_da = df["da_price_gbp_mwh"].astype(float).to_numpy()
    p_rt = df["rt_price_gbp_mwh"].astype(float).to_numpy()
    return p_da, p_rt


def bootstrap_rt_scenarios(p_da: np.ndarray, p_rt_hist: np.ndarray, S: int, seed: int = 42, block_len: int = 48):
    """
    Create S RT scenarios by bootstrapping historical spread blocks.
    spread = RT - DA
    Each scenario samples blocks of length block_len (default 1 day) with replacement.
    """
    rng = np.random.default_rng(seed)
    T = len(p_da)
    spread = p_rt_hist - p_da

    n_blocks = int(np.ceil(T / block_len))
    max_start = T - block_len
    if max_start < 0:
        raise ValueError("T is smaller than block_len; reduce block_len or use more data.")

    scenarios = np.zeros((S, T), dtype=float)
    for s in range(S):
        pieces = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, max_start + 1))
            pieces.append(spread[start : start + block_len])
        spread_s = np.concatenate(pieces)[:T]
        scenarios[s, :] = p_da + spread_s

    return scenarios

