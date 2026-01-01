import csv
import html
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.data_loader import load_prices_30min, bootstrap_rt_scenarios
from src.optimisation_da_only import solve_da_only
from src.optimisation_two_stage import solve_two_stage


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fmt_currency(value: float) -> str:
    return f"{value:,.2f}"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def read_first_row(path: str) -> dict:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, {})


def read_rows(path: str) -> list:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_dashboard():
    run = load_yaml("configs/run.yaml")
    mkt = load_yaml("configs/markets.yaml")
    battery = load_yaml("configs/battery.yaml")

    dt = float(run.get("dt_hours", 0.5))
    S = int(run.get("S", 0))
    seed = int(run.get("seed", 0))

    solve_error = None
    da_metrics = {}
    ts_metrics = {}

    try:
        p_da, p_rt_hist = load_prices_30min()
        T = len(p_da)
        p_rt = bootstrap_rt_scenarios(p_da=p_da, p_rt_hist=p_rt_hist, S=S, seed=seed, block_len=48)

        da_res = solve_da_only(p_da=p_da, dt=dt, battery=battery)
        ts_res = solve_two_stage(p_da=p_da, p_rt=p_rt, dt=dt, battery=battery)

        pnl = ts_res["scenario_pnl"]
        da_metrics = {
            "objective": float(da_res["objective"]),
            "avg_ch": float(da_res["ch"].mean()),
            "avg_dis": float(da_res["dis"].mean()),
            "soc_min": float(da_res["soc"].min()),
            "soc_max": float(da_res["soc"].max()),
            "status": da_res["status"],
        }

        ts_metrics = {
            "expected": float(pnl.mean()),
            "p5": float(np.percentile(pnl, 5)),
            "p50": float(np.percentile(pnl, 50)),
            "p95": float(np.percentile(pnl, 95)),
            "best": float(pnl.max()),
            "worst": float(pnl.min()),
        }

    except Exception as exc:
        solve_error = str(exc)
        T = int(run.get("T", 0))

    diagnostics_path = "outputs/tables/scenario_diagnostics.csv"
    diagnostics = read_first_row(diagnostics_path) if os.path.exists(diagnostics_path) else {}

    stress_path = "outputs/tables/stress_test_results.csv"
    stress_rows = read_rows(stress_path) if os.path.exists(stress_path) else []

    top_regime = {}
    worst_regime = {}
    if stress_rows:
        def to_float(row, key):
            try:
                return float(row.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0

        stress_rows = sorted(stress_rows, key=lambda r: to_float(r, "OptionValue"), reverse=True)
        top_regime = stress_rows[0]
        worst_regime = stress_rows[-1]

    charts = [
        ("Two-stage P&L distribution", "charts/pnl_distribution.png"),
        ("DA + RT price scenarios", "charts/price_scenarios.png"),
        ("DA schedule (two-stage)", "charts/da_schedule.png"),
        ("Option value vs spike probability", "charts/option_value_vs_spike_prob.png"),
        ("Option value distribution", "charts/option_value_distribution.png"),
        ("Option value vs DA volatility", "charts/option_value_vs_da_vol.png"),
    ]
    heatmaps = sorted(Path("outputs/charts").glob("option_value_heatmap_spike_*.png"))
    for path in heatmaps:
        charts.append((f"Heatmap: {path.stem.replace('_', ' ')}", f"charts/{path.name}"))

    summary_path = "outputs/summaries/executive_summary.txt"
    summary_text = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_text = f.read().strip()

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def chip(label: str, value: str) -> str:
        return f"<div class='chip'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"

    def stat(label: str, value: str) -> str:
        return f"<div class='stat'><div class='stat-label'>{html.escape(label)}</div><div class='stat-value'>{html.escape(value)}</div></div>"

    diagnostics_block = ""
    if diagnostics:
        diagnostics_block = "\n".join([
            stat("T", diagnostics.get("T", "")),
            stat("S", diagnostics.get("S", "")),
            stat("DA mean", fmt_float(float(diagnostics.get("DA_mean", 0.0)), 2)),
            stat("DA std", fmt_float(float(diagnostics.get("DA_std", 0.0)), 2)),
            stat("RT mean", fmt_float(float(diagnostics.get("RT_mean", 0.0)), 2)),
            stat("RT std", fmt_float(float(diagnostics.get("RT_std", 0.0)), 2)),
            stat("Spread mean", fmt_float(float(diagnostics.get("Spread_mean", 0.0)), 2)),
            stat("Spread std", fmt_float(float(diagnostics.get("Spread_std", 0.0)), 2)),
            stat("Spread p01", fmt_float(float(diagnostics.get("Spread_p01", 0.0)), 2)),
            stat("Spread p99", fmt_float(float(diagnostics.get("Spread_p99", 0.0)), 2)),
            stat("Extreme spread rate", fmt_float(float(diagnostics.get("Extreme_spread_rate", 0.0)) * 100.0, 2) + "%"),
        ])
    else:
        diagnostics_block = "<div class='empty'>Run src.diagnostics to generate scenario diagnostics.</div>"

    stress_block = ""
    if stress_rows:
        top_cells = [
            chip("da_vol", top_regime.get("da_vol", "")),
            chip("rt_noise_vol", top_regime.get("rt_noise_vol", "")),
            chip("spike_prob", top_regime.get("spike_prob", "")),
            chip("OptionValue", fmt_currency(float(top_regime.get("OptionValue", 0.0)))),
        ]
        worst_cells = [
            chip("da_vol", worst_regime.get("da_vol", "")),
            chip("rt_noise_vol", worst_regime.get("rt_noise_vol", "")),
            chip("spike_prob", worst_regime.get("spike_prob", "")),
            chip("OptionValue", fmt_currency(float(worst_regime.get("OptionValue", 0.0)))),
        ]
        stress_block = f"""
        <div class="subgrid">
          <div class="card">
            <h4>Best regime (highest option value)</h4>
            <div class="chip-grid">{''.join(top_cells)}</div>
          </div>
          <div class="card">
            <h4>Worst regime (lowest option value)</h4>
            <div class="chip-grid">{''.join(worst_cells)}</div>
          </div>
        </div>
        """
    else:
        stress_block = "<div class='empty'>Run src.stress_test to populate stress_test_results.csv.</div>"

    if solve_error:
        kpi_block = f"<div class='empty'>Metrics unavailable: {html.escape(solve_error)}</div>"
        da_block = ""
        ts_block = ""
    else:
        option_value = ts_metrics["expected"] - da_metrics["objective"]
        kpi_block = f"""
        <div class="kpi">
          <div class="kpi-label">DA-only P&L</div>
          <div class="kpi-value">£{fmt_currency(da_metrics['objective'])}</div>
          <div class="kpi-foot">Status: {html.escape(da_metrics['status'])}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Two-stage expected P&L</div>
          <div class="kpi-value">£{fmt_currency(ts_metrics['expected'])}</div>
          <div class="kpi-foot">P&L p5/p50/p95: {fmt_currency(ts_metrics['p5'])} / {fmt_currency(ts_metrics['p50'])} / {fmt_currency(ts_metrics['p95'])}</div>
        </div>
        <div class="kpi highlight">
          <div class="kpi-label">Option value</div>
          <div class="kpi-value">£{fmt_currency(option_value)}</div>
          <div class="kpi-foot">Best/Worst: {fmt_currency(ts_metrics['best'])} / {fmt_currency(ts_metrics['worst'])}</div>
        </div>
        """
        da_block = "\n".join([
            stat("Avg charge (MW)", fmt_float(da_metrics["avg_ch"], 3)),
            stat("Avg discharge (MW)", fmt_float(da_metrics["avg_dis"], 3)),
            stat("SoC min", fmt_float(da_metrics["soc_min"], 2)),
            stat("SoC max", fmt_float(da_metrics["soc_max"], 2)),
        ])
        ts_block = "\n".join([
            stat("Scenarios (S)", str(S)),
            stat("Time steps (T)", str(T)),
            stat("Delta t (hours)", fmt_float(dt, 2)),
            stat("Bootstrap block (periods)", "48"),
        ])

    chart_cards = []
    for title, rel_path in charts:
        img_path = Path("outputs") / rel_path
        if img_path.exists():
            chart_cards.append(
                f"<div class='card chart'><h4>{html.escape(title)}</h4><img src='{rel_path}' alt='{html.escape(title)}'></div>"
            )
        else:
            chart_cards.append(
                f"<div class='card chart missing'><h4>{html.escape(title)}</h4><div class='empty'>Missing: {html.escape(str(img_path))}</div></div>"
            )

    summary_block = ""
    if summary_text:
        summary_block = f"<pre class='summary'>{html.escape(summary_text)}</pre>"
    else:
        summary_block = "<div class='empty'>Run src.summarise after stress_test to generate an executive summary.</div>"

    html_out = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Battery Multi-Market Trading Dashboard</title>
  <style>
    :root {{
      --bg: #f5f1ea;
      --ink: #121212;
      --muted: #5c5c5c;
      --card: #ffffff;
      --accent: #f5a623;
      --accent-2: #3a6ea5;
      --line: #e1ded6;
      --shadow: 0 18px 45px rgba(18, 18, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Avenir", "Gill Sans", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background: var(--bg);
      min-height: 100vh;
      position: relative;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: -20% -10% auto auto;
      width: 60vw;
      height: 60vw;
      background: radial-gradient(circle, rgba(245, 166, 35, 0.35), transparent 60%);
      z-index: 0;
    }}
    body::after {{
      content: "";
      position: fixed;
      inset: auto auto -30% -20%;
      width: 70vw;
      height: 70vw;
      background: radial-gradient(circle, rgba(58, 110, 165, 0.25), transparent 60%);
      z-index: 0;
    }}
    .wrap {{
      position: relative;
      z-index: 1;
      max-width: 1200px;
      margin: 0 auto;
      padding: 48px 24px 72px;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 32px;
    }}
    h1 {{
      font-family: "Georgia", "Times New Roman", serif;
      font-size: clamp(28px, 4vw, 42px);
      margin: 0;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    section {{
      margin-bottom: 36px;
    }}
    h2 {{
      font-size: 18px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 16px;
    }}
    h4 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .card {{
      background: var(--card);
      border-radius: 18px;
      padding: 18px;
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .kpi {{
      background: var(--card);
      border-radius: 18px;
      padding: 20px;
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .kpi.highlight {{
      border-color: var(--accent);
      box-shadow: 0 20px 50px rgba(245, 166, 35, 0.2);
    }}
    .kpi-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .kpi-value {{
      font-size: 28px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .kpi-foot {{
      font-size: 12px;
      color: var(--muted);
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .stat {{
      background: #fdfbf7;
      border-radius: 14px;
      padding: 12px 14px;
      border: 1px solid var(--line);
    }}
    .stat-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .stat-value {{
      font-size: 18px;
      margin-top: 4px;
    }}
    .chip-grid {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }}
    .chip {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      background: #f8f3ea;
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      font-size: 13px;
    }}
    .chip span {{
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 10px;
    }}
    .chip strong {{
      font-size: 16px;
    }}
    .subgrid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .chart img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .missing {{
      border-style: dashed;
    }}
    .empty {{
      padding: 12px;
      border-radius: 12px;
      background: #f7f2eb;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary {{
      white-space: pre-wrap;
      background: #fff9f0;
      border-radius: 16px;
      padding: 16px;
      border: 1px solid var(--line);
      font-size: 13px;
      line-height: 1.5;
    }}
    .footer {{
      font-size: 12px;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div class="subtitle">Battery Multi-Market Trading Optimisation</div>
        <h1>Operations & Risk Dashboard</h1>
      </div>
      <div class="footer">
        <div>Generated: {html.escape(generated)}</div>
        <div>Dataset: data/processed/prices_30min.csv</div>
      </div>
    </header>

    <section>
      <h2>Run Snapshot</h2>
      <div class="grid">
        {kpi_block}
      </div>
    </section>

    <section>
      <h2>Model Inputs</h2>
      <div class="grid">
        <div class="card">
          <h4>Run configuration</h4>
          <div class="stat-grid">
            {stat("T", str(T))}
            {stat("S", str(S))}
            {stat("dt (hours)", fmt_float(dt, 2))}
            {stat("seed", str(seed))}
          </div>
        </div>
        <div class="card">
          <h4>Battery parameters</h4>
          <div class="stat-grid">
            {stat("E_max (MWh)", fmt_float(float(battery.get("E_max", 0.0)), 2))}
            {stat("P_max (MW)", fmt_float(float(battery.get("P_max", 0.0)), 2))}
            {stat("eta_c", fmt_float(float(battery.get("eta_c", 0.0)), 3))}
            {stat("eta_d", fmt_float(float(battery.get("eta_d", 0.0)), 3))}
            {stat("SoC0 (MWh)", fmt_float(float(battery.get("SoC0", 0.0)), 2))}
          </div>
        </div>
        <div class="card">
          <h4>Market assumptions</h4>
          <div class="stat-grid">
            {stat("DA base price", fmt_float(float(mkt.get("da_base_price", 0.0)), 2))}
            {stat("DA vol", fmt_float(float(mkt.get("da_vol", 0.0)), 2))}
            {stat("RT noise vol", fmt_float(float(mkt.get("rt_noise_vol", 0.0)), 2))}
            {stat("Spike prob", fmt_float(float(mkt.get("spike_prob", 0.0)), 3))}
            {stat("Spike size", fmt_float(float(mkt.get("spike_size", 0.0)), 2))}
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2>Dispatch Diagnostics</h2>
      <div class="grid">
        <div class="card">
          <h4>DA-only behavior</h4>
          <div class="stat-grid">
            {da_block if da_block else "<div class='empty'>DA-only metrics not available.</div>"}
          </div>
        </div>
        <div class="card">
          <h4>Scenario construction</h4>
          <div class="stat-grid">
            {ts_block if ts_block else "<div class='empty'>Scenario metrics not available.</div>"}
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2>Scenario Diagnostics</h2>
      <div class="card">
        <div class="stat-grid">
          {diagnostics_block}
        </div>
      </div>
    </section>

    <section>
      <h2>Stress Test Summary</h2>
      {stress_block}
    </section>

    <section>
      <h2>Charts</h2>
      <div class="grid">
        {"".join(chart_cards)}
      </div>
    </section>

    <section>
      <h2>Executive Summary</h2>
      {summary_block}
    </section>

  </div>
</body>
</html>
""".strip()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dashboard.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    build_dashboard()


if __name__ == "__main__":
    main()
