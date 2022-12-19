import random
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from .decision_tree import Node
from .split_functions import split_information_gain


class RandomForestDT(object):
    def __init__(self, df: pd.DataFrame, attributes, labels, k):
        self.df = df
        self.attributes = attributes
        self.labels = labels
        self.tree: Node = rf_id3(self.df, self.attributes, k)

    def predict(self, test_example) -> Optional[str]:
        """Make a prediction for a given test example."""
        node = self.tree
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


class RandomForest(object):
    def __init__(self, df: pd.DataFrame, attributes, labels, m, num_attr):
        self.df = df.sample(frac=1, replace=True)
        self.attributes = attributes
        self.labels = labels
        self.trees = [RandomForestDT(self.df, attributes, labels, num_attr) for _ in range(m)]

    def predict(self, x) -> str:
        predictions = Counter([tree.predict(x) for tree in self.trees])
        return max(predictions, key=predictions.get)


def rf_id3(s: pd.DataFrame, attributes: dict, num_attr, label="label") -> Node:
    """
    Construct a decision tree using the id3 algorithm. If a negative depth is
    supplied, then the algorithm will construct a tree with the maximum
    possible depth allowed by the call stack.
    """
    root = Node(s)
    if (s[label] == s[label].iloc[0]).all():
        root.label = s[label].iloc[0]
        return root
    if num_attr > len(attributes):
        root.label = s[label].value_counts().index[0]
        return root
    sample_attributes = random.sample(attributes.keys(), k=num_attr)
    a: str = split_information_gain(s, {k: attributes[k] for k in sample_attributes})
    root.attribute = a
    for v in attributes[a]:
        subset = s[s[a] == v]
        if subset.empty:
            root.label = s[label].value_counts().index[0]
        else:
            new_attributes = {k: v for k, v in attributes.items() if k != a}
            child = rf_id3(subset, new_attributes, num_attr=num_attr)
            root.children.append(child)
    return root
