"""main.py is the driver for the ID3 algorithm"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from . import split_functions
from .adaboost import AdaBoost
from .build_features import build_features
from .decision_tree import DecisionTree


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

    @staticmethod
    def question2b():
        dataset = "bank"
        print("2b")
        df, attributes, labels = build_features(dataset)
        df["_weight"] = np.ones(len(df.index)) / len(df.index)
        results_table = [
            ["mode", "m", "accuracy"],
        ]
        for m in range(500):
            print(f"{m=}")
            trees = [DecisionTree(df.sample(frac=0.05, replace=True), attributes, labels) for _ in range(m + 1)]
            for mode in ("test", "training"):
                print(f"{mode=}")
                test_df, _, _ = build_features(dataset=dataset, test=mode == "test")
                errors = 0
                for j in test_df.index:
                    predictions = Counter([tree.predict(test_df.iloc[j]) for tree in trees])
                    prediction = max(predictions, key=predictions.get)
                    y = test_df["label"][j]
                    errors += prediction != y
                accuracy = 1 - (errors / len(test_df))
                print(f"{accuracy=}")
                results_table.append([mode, str(m), str(accuracy)])
        with open("reports/2b.txt", "w+") as f:
            for row in results_table:
                f.write(",".join(row) + "\n")

    @staticmethod
    def question2ci():
        dataset = "bank"
        df, attributes, labels = build_features(dataset)
        df["_weight"] = np.ones(len(df.index)) / len(df.index)
        results_table = [
            ["i", "simple/bagged", "bias", "variance", "gen_error"],
        ]
        predictors = []
        m = 100
        n = 500
        for _ in range(m):
            sample = df.sample(n=1000, replace=False)
            predictors.append(
                [DecisionTree(sample.sample(frac=0.05, replace=True), attributes, labels) for _ in range(n)]
            )
        simple_predictors = [predictor[0] for predictor in predictors]
        to_int = lambda x: 1 if x else -1
        test_df, _, _ = build_features(dataset=dataset, test=True)
        for i in test_df.index:
            predictions = tuple(map(to_int, [tree.predict(test_df.iloc[i]) for tree in simple_predictors]))
            hmean = sum(predictions) / len(predictions)
            offset = to_int(test_df["label"].iloc[i])
            bias = (hmean - offset) ** 2
            s = 1 / (len(predictions) - 1) * sum((p - hmean) ** 2 for p in predictions)
            gen_error = bias + s
            results_table.append(list(map(str, [i, "simple", round(bias, 4), s, gen_error])))
            # y = test_df["label"][j]
            # errors += prediction != y
        with open("reports/2ci.csv", "w+") as f:
            for row in results_table:
                f.write(",".join(row) + "\n")
        """
        for m in range(500):
            for mode in ("test", "training"):
                errors = 0
                for j in test_df.index:
                    predictions = Counter([tree.predict(test_df.iloc[j]) for tree in trees])
                    prediction = max(predictions, key=predictions.get)
                    y = test_df["label"][j]
                    errors += prediction != y
                accuracy = 1 - (errors / len(test_df))
                results_table.append([mode, str(m), str(accuracy)])
        with open("reports/2c.txt", "w+") as f:
            for row in results_table:
                f.write(",".join(row) + "\n")
            """


if __name__ == "__main__":
    """
    The following code runs each experiment from the homework. It will likely
    take over an hour to run all experiments (subject to hardware limitations).
    """
    # StatsQuest.heart_disease()
    parser = argparse.ArgumentParser(description="Runner for ensemble learner experiments")
    parser.add_argument("--q2b", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        print("Running all experiments (this is going to take awhile)...")
        HW2.question2a()
        HW2.question2b()
        HW2.question2ci()
    if args.q2b:
        HW2.question2b()
