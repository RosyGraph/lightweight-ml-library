import pandas as pd
from fractions import Fraction
from build_features import build_features
from split_functions import *


class DecisionTree(object):
    def __init__(self, df: pd.DataFrame, attributes, labels, split_func=split_information_gain):
        self.df = df
        self.attributes = attributes
        self.labels = labels
        self.tree = id3(self.df, self.attributes, split_func=split_func)

    def predict(self, test_example):
        node = self.tree
        test_attr_value = test_example[node.attribute]
        while node.children:
            a0 = node.attribute
            test_a0_value = test_example[a0]
            favorite_child_score = 0
            favorite_child = node.children[0]
            for child in node.children:
                c = child.s.loc[child.s[a0] == test_a0_value, a0].count()
                if c > favorite_child_score:
                    favorite_child_score = c
                    favorite_child = child
            node = favorite_child
        return node.label

    def test_accuracy(self, test_df, label="label"):
        return sum(self.predict(row) == row[label] for _, row in test_df.iterrows()) / len(test_df)


class Node(object):
    def __init__(self, s):
        self.s = s
        self.label = None
        self.children = []
        self.attribute = None


def id3(s, attributes, label="label", split_func=split_information_gain, depth=6):
    root = Node(s)
    if (s[label] == s[label].iloc[0]).all():
        root.label = s[label].iloc[0]
        return root
    if not attributes or depth == 0:
        root.label = s[label].value_counts().index[0]
        return root
    a = split_func(s, attributes)
    root.attribute = a
    for v in attributes[a]:
        subset = s[s[a] == v]
        if subset.empty:
            root.label = s[label].value_counts().index[0]
        else:
            new_attributes = {k: v for k, v in attributes.items() if k != a}
            child = id3(subset, new_attributes, label, split_func, depth=depth - 1)
            root.children.append(child)
    return root
