import yaml
import numpy as np
import pulp

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def solve_da_only(p_da: np.ndarray, dt: float, battery: dict):
    """
    Deterministic DA-only battery arbitrage.
    No RT adjustments. Single scenario.
    Returns dict with schedules, SoC, objective value.
    """
    T = len(p_da)

    E_max = float(battery["E_max"])
    P_max = float(battery["P_max"])
    eta_c = float(battery["eta_c"])
    eta_d = float(battery["eta_d"])
    soc0 = float(battery["SoC0"])

    # Model
    m = pulp.LpProblem("battery_DA_only", pulp.LpMaximize)

    ch = pulp.LpVariable.dicts("ch", range(T), lowBound=0)
    dis = pulp.LpVariable.dicts("dis", range(T), lowBound=0)
    soc = pulp.LpVariable.dicts("soc", range(T + 1), lowBound=0, upBound=E_max)

    # Initial SoC
    m += soc[0] == soc0

    # Constraints
    for t in range(T):
        m += ch[t] <= P_max
        m += dis[t] <= P_max

        # SoC dynamics
        m += soc[t + 1] == soc[t] + eta_c * ch[t] * dt - (dis[t] * dt) / eta_d

    # Objective: profit = sum price * (dis - ch) * dt
    m += pulp.lpSum(p_da[t] * (dis[t] - ch[t]) * dt for t in range(T))

    # Solve
    status = m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"DA-only optimisation did not solve optimally: {pulp.LpStatus[status]}")

    ch_sol = np.array([pulp.value(ch[t]) for t in range(T)])
    dis_sol = np.array([pulp.value(dis[t]) for t in range(T)])
    soc_sol = np.array([pulp.value(soc[t]) for t in range(T + 1)])
    obj = float(pulp.value(m.objective))

    return {
        "ch": ch_sol,
        "dis": dis_sol,
        "soc": soc_sol,
        "objective": obj,
        "status": pulp.LpStatus[status],
    }


def main():
    run = load_yaml("configs/run.yaml")
    mkt = load_yaml("configs/markets.yaml")
    battery = load_yaml("configs/battery.yaml")

    dt = float(run["dt_hours"])
    T = int(run["T"])
    seed = int(run["seed"])

    # Import scenario generation from your package
    from src.data_loader import load_prices_30min

    p_da, _ = load_prices_30min()
    T = len(p_da)

    res = solve_da_only(p_da=p_da, dt=dt, battery=battery)

    print("DA-only status:", res["status"])
    print("DA-only objective (profit):", round(res["objective"], 2))
    print("Avg charge MW:", round(res["ch"].mean(), 3), "Avg discharge MW:", round(res["dis"].mean(), 3))
    print("SoC min/max:", round(res["soc"].min(), 3), round(res["soc"].max(), 3))


if __name__ == "__main__":
    main()
