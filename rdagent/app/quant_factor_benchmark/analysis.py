
import json
from pathlib import Path
import pickle
import pandas as pd

from rdagent.app.quant_factor_benchmark.design.benchmark import summarize_res as summarize_res
from rdagent.components.benchmark.conf import BenchmarkSettings


results = {
    "1 round experiment": "git_ignore_folder/eval_results/res_promptV220240724-060037.pkl",
}

# Get index map from the json file
index_map = {}
bs = BenchmarkSettings()
def load(json_file_path: Path) -> None:
    with open(json_file_path, "r") as file:
        factor_dict = json.load(file)
    for factor_name, factor_data in factor_dict.items():
        index_map[factor_name] = (factor_name, factor_data["Category"], factor_data["Difficulty"])

load(bs.bench_data_path)

def load_data(file_path):
    file_path = Path(file_path)

    if not (file_path.is_file() and file_path.suffix == ".pkl" and file_path.name.startswith("res_")):
        raise ValueError("You may get a invalid file path")

    sum_df = []
    with file_path.open("rb") as f:
        res = pickle.load(f)

    sum_df.append(summarize_res(res))
    sum_df = pd.concat(sum_df, axis=1)
    if sum_df.shape[0] == 0:
        raise ValueError("No data in the file")
    print(file_path, sum_df.shape)

    index = [
        "FactorSingleColumnEvaluator",
        "FactorOutputFormatEvaluator",
        "FactorRowCountEvaluator",
        "FactorIndexEvaluator",
        "FactorMissingValuesEvaluator",
        "FactorEqualValueCountEvaluator",
        "FactorCorrelationEvaluator",
        "run factor error",
    ]

    # reindex in case of failing to run evaluator.
    # If all implemented factors fail to run, some evaluators may not appear in the result.
    sum_df = sum_df.reindex(index, axis=0)

    sum_df_clean = sum_df.T.groupby(level=0).apply(lambda x: x.reset_index(drop=True))

    # sum_df.columns

    def get_run_error(sum_df_clean):
        run_error = sum_df_clean["run factor error"]

        run_error = run_error.unstack()

        run_error = run_error.T.fillna(False).astype(bool)  # null indicate no exception

        succ_rate = ~run_error
        succ_rate = succ_rate.mean(axis=0).to_frame("success rate")
        # make it display in a percentage rate
        # succ_rate["success rate"] = succ_rate["success rate"].map(lambda x: f"{x:.2%}")
        return succ_rate

    succ_rate = get_run_error(sum_df_clean)

    def reformat_succ_rate(display_df):
        """You may get dataframe like this:

                                    success rate
        250-day_high_distance             80.00%
        Corr_Close_Turnover               20.00%
        EP_TTM                            20.00%
        High_Frequency_Skewness           60.00%
        Momentum                          50.00%
        Morning_30-min_Return             30.00%
        UID                                0.00%
        Weighted_Earnings_Frequency       10.00%
        """

        new_idx = []
        display_df = display_df[display_df.index.isin(index_map.keys())]
        # display_df = display_df.reindex(index_map.keys())
        for idx in display_df.index:
            new_idx.append(index_map[idx])
        display_df.index = pd.MultiIndex.from_tuples(
            new_idx,
            names=["Factor", "Category", "Difficulty"],
        )

        display_df = display_df.swaplevel(0, 2).swaplevel(0, 1).sort_index(axis=0)

        def sort_key_func(x):
            order_v = []
            for i in x:
                order_v.append({"Easy": 0, "Medium": 1, "Hard": 2, "New Discovery": 3}.get(i, i))
            return order_v

        return display_df.sort_index(key=sort_key_func)

    succ_rate_f = reformat_succ_rate(succ_rate)
    succ_rate_f

    sum_df_clean["FactorRowCountEvaluator"]

    def get_run_error(eval_series):
        eval_series = eval_series.unstack()

        succ_rate = eval_series.T.fillna(False).astype(bool)  # false indicate failure

        succ_rate = succ_rate.mean(axis=0).to_frame("success rate")
        # make it display in a percentage rate
        # succ_rate["success rate"] = succ_rate["success rate"].map(lambda x: f"{x:.2%}")
        return succ_rate

    format_issue = (
        sum_df_clean["FactorRowCountEvaluator"] & sum_df_clean["FactorIndexEvaluator"]
    )

    format_succ_rate = get_run_error(format_issue)
    format_succ_rate_f = reformat_succ_rate(format_succ_rate)

    corr = sum_df_clean["FactorCorrelationEvaluator"] * format_issue

    corr = corr.unstack().T.mean(axis=0).to_frame("corr(only success)")
    corr_res = reformat_succ_rate(corr)

    corr_max = sum_df_clean["FactorCorrelationEvaluator"] * format_issue

    corr_max = corr_max.unstack().T.max(axis=0).to_frame("corr(only success)")
    corr_max_res = reformat_succ_rate(corr_max)

    value_max = sum_df_clean["FactorMissingValuesEvaluator"] * format_issue
    value_max = value_max.unstack().T.max(axis=0).to_frame("max_value")
    value_max_res = reformat_succ_rate(value_max)

    value_avg = (
        (sum_df_clean["FactorMissingValuesEvaluator"] * format_issue)
        .unstack()
        .T.mean(axis=0)
        .to_frame("avg_value")
    )
    value_avg_res = reformat_succ_rate(value_avg)

    result_all = pd.concat(
        {
            "avg. Correlation (value only)": corr_res.iloc[:, 0],
            "avg. Format successful rate": format_succ_rate_f.iloc[:, 0],
            "avg. Run successful rate": succ_rate_f.iloc[:, 0],
            "max. Correlation": corr_max_res.iloc[:, 0],
            "max. accuracy": value_max_res.iloc[:, 0],
            "avg. accuracy": value_avg_res.iloc[:, 0],
        },
        axis=1,
    )

    def result_all_key_order(x):
        order_v = []
        for i in x:
            order_v.append(
                {
                    "avg. Run successful rate": 0,
                    "avg. Format successful rate": 1,
                    "avg. Correlation (value only)": 2,
                    "max. Correlation": 3,
                    "max. accuracy": 4,
                    "avg. accuracy": 5,
                }.get(i, i),
            )
        return order_v

    df = result_all.sort_index(axis=1, key=result_all_key_order)
    print(df)

    # Calculate the mean of each column
    mean_values = df.fillna(0.0).mean()
    mean_df = pd.DataFrame(mean_values).T

    # TODO: set it as multi-index
    # Assign the MultiIndex to the DataFrame
    mean_df.index = pd.MultiIndex.from_tuples([("-", "-", "Average")], names=["Factor", "Category", "Difficulty"])

    # Append the mean values to the end of the dataframe
    df_w_mean = pd.concat([df, mean_df]).astype("float")

    return df_w_mean


def display_df(df):
    # This depends on jupyter
    def _single_formatter(column):
        if column.endswith("rate"):

            def _f(x):
                return "{:.2%}".format(x) if pd.notnull(x) else "-"

        else:

            def _f(x):
                return "{:.4}".format(x) if pd.notnull(x) else "-"

        return _f

    def get_formatters():
        # Show NaN or None as '-'.  Don't convert the value to string
        fmts = {column: _single_formatter(column) for column in df.columns}
        return fmts

    # TODO display
    df_w_mean.drop(["max. accuracy", "avg. accuracy"], axis=1).style.format(get_formatters()).background_gradient(
        axis=0, vmax=1, vmin=0, cmap=__import__("seaborn").light_palette("green", as_cmap=True)
    )


final_res = {}
for k, p in results.items():
    df = load_data(p)
    print(df)
    final_res[k] = df.iloc[-1, :]

final_res = pd.DataFrame(final_res)


# TODO plot it with seaborn and save it as a file
final_res.drop(["max. accuracy", "avg. accuracy"], axis=0).T

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["axes.unicode_minus"] = False


def change_fs(font_size):
    font_size = font_size
    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title


change_fs(20)


# Prepare the data for plotting
plot_data = final_res.drop(["max. accuracy", "avg. accuracy"], axis=0).T
plot_data = plot_data.reset_index().melt("index", var_name="a", value_name="b")

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x="index", y="b", hue="a", data=plot_data)

# Set the labels and title
plt.xlabel("Method")
plt.ylabel("Value")
plt.title("Comparison of Different Methods")

# Save the plot as a file
plt.savefig("comparison_plot.png")
