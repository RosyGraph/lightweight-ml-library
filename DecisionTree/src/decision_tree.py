from typing import Optional

import numpy as np
import pandas as pd


class SplitFunctions(object):
    @staticmethod
    def majority_error_split(s: pd.DataFrame, attributes, label="label") -> str:
        """Return the attribute on which to split using majority error."""
        me_dict = dict()
        me = SplitFunctions.majority_error(s, label)
        for attribute in attributes:
            tmp = me
            for value, p in zip(
                s[attribute].value_counts(normalize=True).index, s[attribute].value_counts(normalize=True)
            ):
                subset = s[s[attribute] == value]
                if not subset.empty:
                    tmp -= p * SplitFunctions.majority_error(subset, label)
            me_dict[attribute] = tmp
        return max(me_dict, key=me_dict.get)

    @staticmethod
    def majority_error(s: pd.DataFrame, label="label") -> np.floating:
        """Return the majority error of the subset s."""
        return 1 - s[label].value_counts(normalize=True).iloc[0]

    @staticmethod
    def split_information_gain(s, attributes) -> str:
        """Return the attribute on which to split using information gain."""
        attr_dict = {attribute: SplitFunctions.information_gain(s, attributes, attribute) for attribute in attributes}
        return max(attr_dict, key=attr_dict.get)

    @staticmethod
    def information_gain(s, attributes, attribute) -> float:
        """Return the information gain for a subset s for the given attribute."""
        gain = SplitFunctions.entropy(s)
        for v in attributes[attribute]:
            subset = s.loc[s[attribute] == v]
            gain -= (subset.size / s.size) * SplitFunctions.entropy(subset)
        return gain

    @staticmethod
    def entropy(s, y="label") -> float:
        """Return the entropy of the subset s."""
        proportions = s[y].value_counts(normalize=True)
        return -sum(map(lambda n: n * np.log2(n), proportions))

    @staticmethod
    def gini(s, attributes):
        """Return the attribute on which to split using Gini index."""
        gini_dict = dict()
        for attribute in attributes:
            gini_dict[attribute] = SplitFunctions.gini_impurity(s, attributes, attribute)
        return min(gini_dict, key=gini_dict.get)

    @staticmethod
    def gini_impurity_helper(s, attribute, v):
        subset = s[s[attribute] == v]
        impurity = np.float64(1)
        for c in subset["label"].value_counts(normalize=True):
            impurity -= np.power(c, 2)
        return impurity * np.float64(len(subset) / len(s))

    @staticmethod
    def gini_impurity(s: pd.DataFrame, _, attribute) -> np.floating:
        """Return the gini impurity of the subset s for the given attribute."""
        return sum(map(lambda v: SplitFunctions.gini_impurity_helper(s, attribute, v), s[attribute].unique()))


class DecisionTree(object):
    """The DecisionTree class represents the decision process for labelling data using the ID3 algorithm."""

    def __init__(
        self, df: pd.DataFrame, attributes, labels, split_func=SplitFunctions.split_information_gain, max_depth=-1
    ):
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


def id3(
    s: pd.DataFrame, attributes: dict, label="label", split_func=SplitFunctions.split_information_gain, depth=-1
) -> Node:
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
