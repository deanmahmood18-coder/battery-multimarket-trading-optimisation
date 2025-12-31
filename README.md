# Battery Multi-Market Trading Optimisation  
**Two-Stage Stochastic Strategy for Energy Storage**

---

## Overview
Battery energy storage assets create value by deciding **when** to charge and discharge and **how much capacity** to commit across different electricity markets. In practice, traders must balance:

- Locking in revenue in the **day-ahead (DA)** market  
- Preserving flexibility to capture **real-time (RT) / imbalance-style** price movements  
- Operating within strict physical constraints on power, energy, and efficiency  

This project frames battery trading as a **decision problem under uncertainty** and quantifies the **value of flexibility** using optimisation and regime stress testing.

---

## Problem Definition
A battery operator must decide how much capacity to commit day-ahead versus how much to keep flexible for real-time opportunities when prices are uncertain and may exhibit spikes.

Key challenges:
- DA commitments are fixed in advance
- RT prices are volatile and uncertain
- Battery dispatch is constrained by power limits, energy capacity, efficiency, and state of charge

The objective is to determine **when flexibility creates value** and how that value changes across market regimes.

---

## Baseline: Day-Ahead-Only Optimisation
The starting point is a deterministic DA-only arbitrage model:

- Single price series
- Linear optimisation
- Full battery constraints
- Objective: maximise DA trading profit

This provides a clean benchmark for comparison.

Run:
```bash
python -m src.optimisation_da_only
```
## Model: Two-Stage Stochastic Optimisation

### Structure
The core model is a **two-stage stochastic optimisation**, implemented as a **deterministic-equivalent linear programme**.

---

### Stage 1 – Day-Ahead Decisions (Non-Anticipative)
Shared across all scenarios:

- Day-ahead charge schedule `ch_DA[t]`
- Day-ahead discharge schedule `dis_DA[t]`

These decisions are **fixed before real-time prices are known** and must be identical across all scenarios.

---

### Stage 2 – Real-Time Adjustments (Scenario-Specific)
Once real-time prices realise for scenario `s`:

- Adjustment charge `ch_ADJ[s,t]`
- Adjustment discharge `dis_ADJ[s,t]`

Total dispatch respects battery constraints:

- Power limits
- State-of-charge dynamics
- Charge and discharge efficiency

---

### Objective
Maximise **expected profit across scenarios**:

- Day-ahead energy earns DA prices
- Real-time adjustments earn incremental value `(RT − DA)`

This formulation produces a **full distribution of realised P&L**, rather than a single point estimate.

Run:
python -m src.optimisation_two_stage

## Stress Testing and Regime Analysis

Rather than relying on a single price path, the strategy is evaluated across **synthetic market regimes** that vary:

- Day-ahead volatility  
- Real-time noise (forecast error proxy)  
- Spike probability (imbalance stress proxy)  

For each regime, the model computes:

- DA-only P&L  
- Two-stage expected P&L  
- Distribution of two-stage outcomes (p5 / p50 / p95, worst, best)  
- **Option value of flexibility**

---

### Option Value Definition

\[
\text{Option Value} = \mathbb{E}[\text{Two-Stage P\&L}] - \text{DA-Only P\&L}
\]

---

### How to Run

```bash
python -m src.stress_test
python -m src.plots_stress
python -m src.summarise
```
Outputs
outputs/tables/stress_test_results.csv
outputs/charts/option_value_vs_spike_prob.png
outputs/summaries/executive_summary.txt

## Key Insights

- **Flexibility behaves like an option on volatility.**  
  In calm markets, committing day-ahead captures most of the available value.

- **Optionality becomes valuable in spiky regimes.**  
  As volatility and spike frequency increase, preserving real-time flexibility materially improves expected returns.

- **Risk matters, not just averages.**  
  The two-stage framework produces a full P&L distribution, enabling downside and upside analysis rather than relying on a single forecast.

---

## Project Structure ## 
configs/        Model assumptions and parameters
src/            Scenario generation, optimisation, evaluation
outputs/        Charts, tables, and summaries
tests/          Sanity checks
## How to Run

### Sanity-check scenario generation

python -m src.backtest
- **baseline model**
python -m src.optimisation_da_only
- **two stage eval**
python -m src.optimisation_two_stage
python -m src.evaluation
- **stress test and summaries**
python -m src.stress_test
python -m src.plots_stress
python -m src.summarise




