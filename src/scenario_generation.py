import numpy as np

def generate_da_prices(T: int, base: float, vol: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # simple mean-reverting-ish random walk
    eps = rng.normal(0, vol, size=T)
    p = base + np.cumsum(0.15 * eps)
    return p

def generate_rt_scenarios(p_da: np.ndarray, S: int, noise_vol: float,
                          spike_prob: float, spike_size: float,
                          seed: int = 123) -> np.ndarray:
    """
    RT = DA + noise + occasional spikes
    Returns array shape (S, T)
    """
    rng = np.random.default_rng(seed)
    T = len(p_da)

    noise = rng.normal(0, noise_vol, size=(S, T))

    spikes = rng.random((S, T)) < spike_prob
    spike_sign = rng.choice([-1, 1], size=(S, T))
    spike_vals = spikes * spike_sign * spike_size

    p_rt = p_da[None, :] + noise + spike_vals
    return p_rt