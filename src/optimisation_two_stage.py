import yaml
import numpy as np
import pulp
from src.data_loader import load_prices_30min, bootstrap_rt_scenarios


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def solve_two_stage(p_da: np.ndarray, p_rt: np.ndarray, dt: float, battery: dict):
    """
    Deterministic-equivalent two-stage LP.

    Shared (non-anticipative) DA decisions:
      ch_DA[t], dis_DA[t]

    Scenario-specific RT adjustment decisions:
      ch_ADJ[s,t], dis_ADJ[s,t], SoC[s,t]

    Objective:
      Average over scenarios of:
        DA revenue at DA price
        + adjustment revenue at (RT - DA) spread

    Returns:
      dict with DA schedule, scenario P&L, and summary stats.
    """
    T = len(p_da)
    S = p_rt.shape[0]
    assert p_rt.shape[1] == T, "p_rt must have shape (S, T)"

    E_max = float(battery["E_max"])
    P_max = float(battery["P_max"])
    eta_c = float(battery["eta_c"])
    eta_d = float(battery["eta_d"])
    soc0 = float(battery["SoC0"])

    m = pulp.LpProblem("battery_two_stage", pulp.LpMaximize)

    # Shared DA decisions
    ch_DA = pulp.LpVariable.dicts("ch_DA", range(T), lowBound=0)
    dis_DA = pulp.LpVariable.dicts("dis_DA", range(T), lowBound=0)

    # Scenario decisions
    ch_ADJ = pulp.LpVariable.dicts("ch_ADJ", (range(S), range(T)), lowBound=0)
    dis_ADJ = pulp.LpVariable.dicts("dis_ADJ", (range(S), range(T)), lowBound=0)
    soc = pulp.LpVariable.dicts("soc", (range(S), range(T + 1)), lowBound=0, upBound=E_max)

    # Constraints
    for s in range(S):
        m += soc[s][0] == soc0
        for t in range(T):
            # Total physical power limits
            m += ch_DA[t] + ch_ADJ[s][t] <= P_max
            m += dis_DA[t] + dis_ADJ[s][t] <= P_max

            # SoC dynamics using total dispatch
            m += soc[s][t + 1] == soc[s][t] + eta_c * (ch_DA[t] + ch_ADJ[s][t]) * dt - ((dis_DA[t] + dis_ADJ[s][t]) * dt) / eta_d

    # Objective: average profit over scenarios
    # DA part (same across scenarios)
    da_profit = pulp.lpSum(p_da[t] * (dis_DA[t] - ch_DA[t]) * dt for t in range(T))

    # Adjustment part (scenario-specific spread vs DA)
    adj_profit = pulp.lpSum(
        (p_rt[s, t] - p_da[t]) * (dis_ADJ[s][t] - ch_ADJ[s][t]) * dt
        for s in range(S) for t in range(T)
    ) / float(S)

    m += da_profit + adj_profit

    status = m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Two-stage optimisation did not solve optimally: {pulp.LpStatus[status]}")

    ch_DA_sol = np.array([pulp.value(ch_DA[t]) for t in range(T)])
    dis_DA_sol = np.array([pulp.value(dis_DA[t]) for t in range(T)])

    # Compute scenario P&L using the solved variables
    scenario_pnl = np.zeros(S, dtype=float)
    for s in range(S):
        pnl = 0.0
        for t in range(T):
            da_term = p_da[t] * (dis_DA_sol[t] - ch_DA_sol[t]) * dt
            ch_adj = pulp.value(ch_ADJ[s][t])
            dis_adj = pulp.value(dis_ADJ[s][t])
            adj_term = (p_rt[s, t] - p_da[t]) * (dis_adj - ch_adj) * dt
            pnl += da_term + adj_term
        scenario_pnl[s] = pnl

    return {
        "status": pulp.LpStatus[status],
        "ch_DA": ch_DA_sol,
        "dis_DA": dis_DA_sol,
        "scenario_pnl": scenario_pnl,
        "expected_pnl": float(scenario_pnl.mean()),
        "objective": float(pulp.value(m.objective)),
    }

def main():
    run = load_yaml("configs/run.yaml")
    battery = load_yaml("configs/battery.yaml")

    dt = float(run["dt_hours"])
    S = int(run["S"])
    seed = int(run["seed"])

    # Load REAL DA + SBP prices
    p_da, p_rt_hist = load_prices_30min("data/processed/prices_30min.csv")

    # Build RT scenarios via bootstrapped historical spreads
    p_rt = bootstrap_rt_scenarios(
        p_da=p_da,
        p_rt_hist=p_rt_hist,
        S=S,
        seed=seed,
        block_len=48,
    )

    res = solve_two_stage(
        p_da=p_da,
        p_rt=p_rt,
        dt=dt,
        battery=battery,
    )

    return res



if __name__ == "__main__":
    res = main()
    pnl = np.array(res.get("scenario_pnl", res.get("pnl_scenario", [])), dtype=float)

    if pnl.size > 0:
        p5, p50, p95 = np.percentile(pnl, [5, 50, 95])
        print("P&L percentiles (p5/p50/p95):", f"{p5:.2f}", f"{p50:.2f}", f"{p95:.2f}")
    else:
        print("No scenario P&L array found in result dict. Available keys:", sorted(res.keys()))
