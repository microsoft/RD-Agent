import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def read_all_stat():
    """Read HDF file and return the dataframe."""
    return pd.read_hdf("all_stat.h5", "data")

def plot_metric_improvement_and_distribution_grid(df, save_prefix='metric_summary_grid'):
    """
    For all metrics, produce a grid figure:
      - For each metric, show two plots:
         1. Barplot of mean difference (merge_12h - others) across all combinations.
         2. Violinplot showing the score distribution for merge_12h and comparison methods.
      - Arrange each metric in a row, and two columns (bar, violin).
      - Save as a single figure.
    """
    metrics = df.index.get_level_values(1).unique()
    compare_cols = ["exp1_full", "exp2_full", "exp1_12h", "exp2_12h"]
    merge_col = "merge_12h"
    all_cols = compare_cols + [merge_col]

    n_metrics = len(metrics)
    ncols = 2
    nrows = n_metrics

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * n_metrics))
    if n_metrics == 1:
        axes = axes.reshape((1, 2))  # Make sure axes is 2D array
    for i, metric in enumerate(metrics):
        # Pivot out just this metric
        pivot_df = df.xs(metric, level=1)[all_cols]

        # 1. Heatmap of absolute values for all experiments, highlight cell if merge_12h is best
        best_mask = pivot_df.values == pivot_df.max(axis=1).values[:, None]
        cell_colors = [["yellow" if best else "white" for best in row] for row in best_mask]
        sns.heatmap(
            pivot_df,
            annot=True, fmt=".2f",
            cmap="Blues", cbar=True,
            ax=axes[i, 0],
            linewidths=0.5,
            linecolor="black",
            annot_kws={"weight": "bold"},
            # highlight best cells in yellow
            # Using seaborn's mask is tricky; instead, set facecolors after plotting:
        )
        # Highlight best columns
        heatmap = axes[i, 0].collections[0]
        for y in range(pivot_df.shape[0]):
            for x in range(pivot_df.shape[1]):
                if best_mask[y, x]:
                    heatmap.axes.add_patch(
                        plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='gold', lw=3)
                    )
        axes[i, 0].set_ylabel("Sample")
        axes[i, 0].set_xlabel("Experiment")
        axes[i, 0].set_title(f"{metric} | Absolute Scores")

        # 2. Violinplot of value distributions
        melted = pivot_df.melt(var_name="condition", value_name="score")
        ax_violin = axes[i, 1]
        sns.violinplot(
            x="condition", y="score", hue="condition", data=melted, 
            palette="muted", ax=ax_violin, legend=False, inner="box"
        )
        ax_violin.set_title(f"{metric} | Distribution")
        if i == 0:
            # Only show legend on violin if desired (or can turn it off everywhere)
            ax_violin.legend_.remove() if ax_violin.legend_ else None

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_allmetrics.png")
    plt.close()

def main():
    df = read_all_stat()
    df = df[~df.index.get_level_values(0).str.contains("V03|V04") & ~df.index.get_level_values(0).str.contains("o1-")]
    print(df)
    # Group by the second index level (experiment), and compute mean and std for each metric/experiment/method
    summary = df.groupby(level=1).agg(['mean', 'std'])
    print("\n===== Summary Table: Mean and Std by Experiment/Metric/Method =====")
    print(summary)
    
    combs = df.index.get_level_values(0).unique()
    cases = []
    for cb in combs:
        exp1, exp2 = cb.split("_")
        cases.extend([exp1, exp2])
    cases = list(set(cases))
    import itertools
    perms = list(itertools.permutations(cases))
    stds = {}
    for p in perms:
        names = [f"{p[i]}_{p[i+1]}" for i in range(0, len(p)-1, 2)]
        mask = df.index.get_level_values(0).isin(names)
        idx = list(sorted(df[mask].groupby(level=0).size().index.values))
        if idx == names:
            summary = df[mask].groupby(level=1).agg(['mean', 'std'])
            print("\n===== Summary Table: Mean and Std by Experiment/Metric/Method =====")
            print(f"Found grouping case: {names}")
            print(summary)
            stds[",".join(names)] = (summary.loc[:, ("merge_12h", "std")])
    std_df = pd.DataFrame(stds)
    print(std_df)
    print(std_df.mean(axis=1))
    plot_metric_improvement_and_distribution_grid(df)

if __name__ == "__main__":
    main()

