import pandas as pd
from typing import Optional
from split_functions import *


class DecisionTree(object):
    """The DecisionTree class represents the decision process for labelling data using the ID3 algorithm."""

    def __init__(self, df: pd.DataFrame, attributes, labels, split_func=split_information_gain, max_depth=-1):
        self.df = df
        self.attributes = attributes
        self.labels = labels
        self.tree: Node = id3(self.df, self.attributes, split_func=split_func, depth=max_depth)

    def predict(self, test_example) -> Optional[str]:
        """Make a prediction for a given test example."""
        node: Node = self.tree
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
    """The Node class is for use in constructing a DecisionTree."""

    def __init__(self, s):
        self.s = s
        self.label: Optional[str] = None
        self.children = []
        self.attribute: Optional[str] = None


def id3(s: pd.DataFrame, attributes: dict, label="label", split_func=split_information_gain, depth=-1) -> Node:
    """
    Construct a decision tree using the id3 algorithm. If a negative depth is
    supplied, then the algorithm will construct a tree with the maximum
    possible depth allowed by the call stack.
    """
    root = Node(s)
    if (s[label] == s[label].iloc[0]).all():
        root.label = s[label].iloc[0]
        return root
    if not attributes or depth == 0:
        root.label = s[label].value_counts().index[0]
        return root
    a: str = split_func(s, attributes)
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
