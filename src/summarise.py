import os
import pandas as pd


def main():
    os.makedirs("outputs/summaries", exist_ok=True)

    input_path = "outputs/tables/stress_test_results.csv"
    if not os.path.exists(input_path):
        print(f"Missing input: {input_path}. Run the stress test to generate it.")
        return

    df = pd.read_csv(input_path)

    best = df.sort_values("OptionValue", ascending=False).head(1).iloc[0]
    worst = df.sort_values("OptionValue", ascending=True).head(1).iloc[0]

    lines = []
    lines.append("Executive summary: Two-stage multi-market battery trading optimisation")
    lines.append("")
    lines.append("Key result:")
    lines.append("Option value (flexibility) increases materially as volatility and spikes increase.")
    lines.append("")
    lines.append("Best regime (highest option value):")
    lines.append(best.to_string())
    lines.append("")
    lines.append("Worst regime (lowest option value):")
    lines.append(worst.to_string())
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- In calm markets, committing day-ahead captures most value; optionality adds little.")
    lines.append("- In spiky/volatile markets, preserving flexibility for real-time adjustments is valuable.")
    lines.append("")

    outpath = "outputs/summaries/executive_summary.txt"
    with open(outpath, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
