"""main.py is the driver for the ID3 algorithm"""
import os

import numpy as np
import pandas as pd

from . import split_functions
from .adaboost import AdaBoost
from .build_features import build_features
from .decision_tree import DecisionTree


def report_test_predictions(dataset="car"):
    test_df, _, _ = build_features(dataset, test=True)
    for f in (split_functions.split_information_gain, split_functions.gini, split_functions.majority_error_split):
        print(f"Testing {f.__name__}...")
        dt = DecisionTree(*build_features("car"), split_func=f)
        print(f"Prediction score: {round(dt.test_accuracy(test_df)*100, 4)}%.")


def compare_depth_and_split(dataset: str, max_depth=5) -> list[dict]:
    results_table = []
    for depth in range(max_depth):
        df, attributes, _ = build_features(dataset)
        ab = AdaBoost(df, attributes, t=depth + 1)
        for mode in ("test", "training"):
            test_df, _, _ = build_features(dataset=dataset, test=mode == "test")
            results_table.append((depth, ab.test_accuracy(test_df), ab.errors))
    return results_table


def store_results(dataset: str, results_table: list[dict]):
    repr_filename = os.path.join("reports", "python", f"{dataset}_decision_tree_comparison")
    print(f"Outputting Python repr to file {repr_filename}...", end=" ")
    with open(repr_filename + ".py", "w+") as f:
        print(repr(results_table), file=f)
    print("Done.")
    results_df = pd.DataFrame(results_table)
    print(results_df)
    tex_filename = os.path.join("reports", "tex", f"{dataset}_decision_tree_comparison")
    print(f"Outputting LaTeX table to file {tex_filename}...", end=" ")
    with open(tex_filename + ".tex", "w+") as f:
        print(results_df.style.to_latex(), file=f)
    print("Done.")


class StatsQuest(object):
    @staticmethod
    def heart_disease():
        df = pd.DataFrame(
            {
                "pain": [True, False, True, True, False, False, True, True],
                "art": [True, True, False, True, True, True, False, True],
                "wt": [205, 180, 210, 167, 156, 125, 168, 172],
                "label": [True, True, True, True, False, False, False, False],
            }
        )
        median = 176
        names = [f"<{median}", f"{median}+"]
        bins = [-np.inf, median, np.inf]
        df["wt"] = pd.cut(df["wt"], bins, labels=names)
        attribute_keys = set(df.columns) - {"label"}
        attributes = dict(zip(attribute_keys, map(lambda c: list(df[c].unique()), attribute_keys)))
        t = 5
        adaboost = AdaBoost(df, attributes, t)
        acc = adaboost.test_accuracy(df)
        print(f"{acc=}")


class HW2(object):
    @staticmethod
    def question2a():
        dataset = "bank"
        results_table = compare_depth_and_split(dataset, max_depth=500)
        print(f"{results_table=}")
        store_results(dataset, results_table)


if __name__ == "__main__":
    """
    The following code runs each experiment from the homework. It will likely
    take over an hour to run all experiments (subject to hardware limitations).
    """
    # StatsQuest.heart_disease()
    HW2.question2a()
