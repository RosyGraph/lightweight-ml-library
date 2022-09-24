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


def compare_depth_and_split():
    results_table = []
    for depth in range(1, 7):
        for f in (split_information_gain, gini, majority_error_split):
            dt = DecisionTree(*build_features("car"), split_func=f, max_depth=depth)
            for dataset in ("test", "training"):
                test_df, _, _ = build_features(dataset="car", test=dataset == "test")
                accuracy = dt.test_accuracy(test_df)
                results_table.append(
                    {"depth": depth, "test/training": dataset, "split function": f.__name__, "accuracy": accuracy}
                )
    results_df = pd.DataFrame(results_table)
    print(results_df)
    filename = os.path.join("reports", "car_decision_tree_comparison.tex")
    print(f"Outputting LaTeX table to file {filename}...")
    with open(filename, "w") as f:
        print(results_df.style.to_latex(), file=f)
    return results_df


if __name__ == "__main__":
    compare_depth_and_split()
    print("Script complete.")
