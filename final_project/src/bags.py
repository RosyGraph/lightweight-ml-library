from .decision_tree import DecisionTree


def bdt_bags(df, attributes, labels, t=1):
    trees = [DecisionTree(df.sample(frac=0.5, replace=True), attributes, labels) for _ in range(t)]
