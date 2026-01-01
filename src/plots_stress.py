import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("outputs/charts", exist_ok=True)

    df = pd.read_csv("outputs/tables/stress_test_results.csv")

    # Average across da_vol and rt_noise_vol for each spike_prob
    agg = df.groupby("spike_prob")["OptionValue"].mean().reset_index()

    plt.figure()
    plt.plot(agg["spike_prob"], agg["OptionValue"], marker="o")
    plt.title("Average option value vs spike probability")
    plt.xlabel("Spike probability")
    plt.ylabel("Option value (Two-stage mean P&L - DA-only P&L)")
    outpath = "outputs/charts/option_value_vs_spike_prob.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

    # Option value distribution
    plt.figure()
    plt.hist(df["OptionValue"], bins=20)
    plt.title("Option value distribution across regimes")
    plt.xlabel("Option value")
    plt.ylabel("Count")
    outpath = "outputs/charts/option_value_distribution.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")

    # Average option value by DA volatility
    agg_da = df.groupby("da_vol")["OptionValue"].mean().reset_index()
    plt.figure()
    plt.plot(agg_da["da_vol"], agg_da["OptionValue"], marker="o")
    plt.title("Average option value vs DA volatility")
    plt.xlabel("DA volatility")
    plt.ylabel("Option value")
    outpath = "outputs/charts/option_value_vs_da_vol.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")

    # Heatmaps by spike probability
    spike_probs = sorted(df["spike_prob"].unique())
    for spike in spike_probs:
        sub = df[df["spike_prob"] == spike]
        pivot = sub.pivot_table(
            index="rt_noise_vol",
            columns="da_vol",
            values="OptionValue",
            aggfunc="mean",
        )
        plt.figure(figsize=(6, 4))
        plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.colorbar(label="Option value")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"Option value heatmap (spike_prob={spike})")
        plt.xlabel("DA volatility")
        plt.ylabel("RT noise vol")
        safe_spike = str(spike).replace(".", "p")
        outpath = f"outputs/charts/option_value_heatmap_spike_{safe_spike}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
