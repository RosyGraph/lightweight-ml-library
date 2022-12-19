import os

import numpy as np

from . import linear_regression
from .build_features import build_income, build_income_df, build_income_test
from .random_forest import RandomForest


def write_predictions(filename, predictions):
    with open(os.path.join("reports", filename), "w+") as f:
        for row in predictions:
            f.write(",".join(list(map(str, row))) + "\n")


def gradient_descent():
    X, y = build_income()
    solver = linear_regression.gradient_descent(X, y)
    test_arr = build_income_test()
    test_IDs, test_X = test_arr[:, 0], test_arr[:, 1:]
    predictions = [("ID", "Prediction")] + [(test_ID, int(solver(x))) for test_ID, x in zip(test_IDs, test_X)]
    write_predictions("predictions_gradient_descent.csv", predictions)
    with open("reports/predictions.csv", "w+") as f:
        for row in predictions:
            f.write(",".join(list(map(str, row))) + "\n")


if __name__ == "__main__":
    df, attributes, labels = build_income_df()
    df["_weight"] = np.ones(len(df.index)) / len(df.index)
    df.rename(columns={"Prediction": "label"}, inplace=True)
    rfr = RandomForest(df, attributes, labels, 8, 4)
    test_df, _, _ = build_income_df(test=True)
    test_df = test_df.drop(["ID"], axis=1)
    predictions = [["ID", "Prediction"]]
    for i in test_df.index:
        example = test_df.iloc[i]
        prediction = rfr.predict(example)
        predictions.append(list(map(str, (i + 1, prediction))))
    write_predictions("predictions_rfr.csv", predictions)
