import numpy as np

from .decision_tree import DecisionTree


class DecisionStump(DecisionTree):
    def __init__(self, df, attributes, labels):
        super().__init__(df, attributes, labels, max_depth=1)

    def error(self, df):
        total_error = np.finfo(float).eps
        for i in df.index:
            prediction = self.predict(df.iloc[i])
            if prediction != df["label"].iloc[i]:
                total_error += df["_weight"][i]
        return total_error


class AdaBoost(object):
    def __init__(self, df, attributes, t):
        self.df = df
        df["_weight"] = np.ones(len(df.index)) / len(df.index)
        self.attibutes = attributes
        self.t = t
        self.votes, self.errors = adaboost(df, attributes, t)

    def predict(self, example):
        predictions = {l: 0 for l in self.df["label"].unique()}
        for tree, wt in self.votes.items():
            prediction = tree.predict(example)
            predictions[prediction] += wt
        return max(predictions, key=predictions.get)

    def test_accuracy(self, dataset):
        correct_ids = 0
        for i in dataset.index:
            prediction = self.predict(dataset.iloc[i])
            if prediction == dataset["label"].iloc[i]:
                correct_ids += 1
        return correct_ids / len(dataset)


def update_weight(row, stump, a):
    if stump.predict(row) == row["label"]:
        return row["_weight"] * np.exp(-a)
    return row["_weight"] * np.exp(a)


def adaboost(df, attributes, t):
    votes = dict()
    errors = []
    for i in range(t):
        stump = DecisionStump(df, attributes, df["label"].unique())
        error = stump.error(df)
        errors.append((i, error))
        a = 0.5 * np.log((1 - error) / error)
        votes[stump] = a
        _update_weight = lambda r: update_weight(r, stump, a)
        df.apply(_update_weight, axis=1)
        df["_weight"] /= df["_weight"].sum()
    return votes, errors
