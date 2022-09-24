"""main.py is the driver for the ID3 algorithm"""
import os
import pandas as pd
from decision_tree import DecisionTree
from build_features import build_features
from split_functions import split_information_gain, gini, majority_error_split


def report_test_predictions(dataset="car"):
    test_df, _, _ = build_features(dataset, test=True)
    for f in (split_information_gain, gini, majority_error_split):
        print(f"Testing {f.__name__}...")
        dt = DecisionTree(*build_features("car"), split_func=f)
        print(f"Prediction score: {round(dt.test_accuracy(test_df)*100, 4)}%.")


def compare_depth_and_split(dataset: str, max_depth=6) -> list[dict]:
    print(f"Comparing split functions for {dataset} for max depth {max_depth} (this may take awhile)...")
    results_table = []
    for depth in range(1, max_depth + 1):
        print(f"Reached depth {depth}.")
        for f in (split_information_gain, gini, majority_error_split):
            dt = DecisionTree(*build_features(dataset), split_func=f, max_depth=depth)
            for mode in ("test", "training"):
                test_df, _, _ = build_features(dataset=dataset, test=mode == "test")
                accuracy = dt.test_accuracy(test_df)
                results_table.append(
                    {"depth": depth, "test/training": mode, "split function": f.__name__, "accuracy": accuracy}
                )
    return results_table


def store_results(dataset: str, results_table: list[dict]):
    repr_filename = os.path.join("reports", "python", f"{dataset}_decision_tree_comparison")
    print(f"Outputting Python repr to file {repr_filename}...", end=" ")
    with open(repr_filename + ".py", "w") as f:
        print(repr(results_table), file=f)
    print("Done.")
    results_df = pd.DataFrame(results_table)
    print(results_df)
    tex_filename = os.path.join("reports", "tex", f"{dataset}_decision_tree_comparison")
    print(f"Outputting LaTeX table to file {tex_filename}...", end=" ")
    with open(tex_filename + ".tex", "w") as f:
        print(results_df.style.to_latex(), file=f)
    print("Done.")


if __name__ == "__main__":
    dataset = "bank"
    results_table = compare_depth_and_split(dataset, max_depth=16)
    store_results(dataset, results_table)
