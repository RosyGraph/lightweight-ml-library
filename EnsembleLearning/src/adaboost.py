import numpy as np

from . import split_functions
from .decision_tree import DecisionTree


class DecisionStump(DecisionTree):
    def __init__(self, df, attributes, labels, w):
        self.w = w
        super().__init__(df, attributes, labels, max_depth=1)

    def error(self, df, w):
        total_error = np.finfo(float).eps
        for i in df.index:
            prediction = self.predict(df.iloc[i])
            if prediction != df["label"].iloc[i]:
                total_error += w[i]
        return total_error


class AdaBoost(object):
    def __init__(self, df, attributes, t):
        self.df = df
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


def adaboost(df, attributes, t):
    votes = dict()
    w = np.ones(len(df)) / len(df)
    errors = []
    for i in range(t):
        stump = DecisionStump(df, attributes, df["label"].unique(), w)
        error = stump.error(df, w)
        errors.append((i, error))
        a = 0.5 * np.log((1 - error) / error)
        votes[stump] = a
        for j in df.index:
            predicted_correctly = stump.predict(df.iloc[j]) == df["label"].iloc[j]
            if predicted_correctly:
                w[j] *= np.exp(-a)
            else:
                w[j] *= np.exp(a)
        w = w / w.sum()
    return votes, errors
