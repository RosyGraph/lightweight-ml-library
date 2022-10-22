import numpy as np
import pandas as pd


def majority_error_split(s: pd.DataFrame, attributes, label="label") -> str:
    """Return the attribute on which to split using majority error."""
    me_dict = dict()
    me = majority_error(s, label)
    for attribute in attributes:
        tmp = me
        for value, p in zip(s[attribute].value_counts(normalize=True).index, s[attribute].value_counts(normalize=True)):
            subset = s[s[attribute] == value]
            if not subset.empty:
                tmp -= p * majority_error(subset, label)
        me_dict[attribute] = tmp
    return max(me_dict, key=me_dict.get)


def majority_error(s: pd.DataFrame, label="label") -> np.floating:
    """Return the majority error of the subset s."""
    return 1 - s[label].value_counts(normalize=True).iloc[0]


def split_information_gain(s, attributes, weights) -> str:
    """Return the attribute on which to split using information gain."""
    attr_dict = {attribute: information_gain(s, attributes, attribute, weights) for attribute in attributes}
    split_attr = max(attr_dict, key=attr_dict.get)
    return split_attr


def information_gain(s, attributes, attribute, weights) -> float:
    """Return the information gain for a subset s for the given attribute."""
    gain = entropy(s, weights)
    for v in attributes[attribute]:
        subset = s.loc[s[attribute] == v]
        gain -= (subset.size / s.size) * entropy(subset, weights)
    return gain


def entropy(s, weights, y="label") -> float:
    """Return the entropy of the subset s."""
    positives = []
    negatives = []
    for i in s.index:
        if s[y][i]:
            positives.append(weights[i])
        else:
            negatives.append(weights[i])
    proportions = np.array([sum(positives), sum(negatives)])
    proportions /= sum(proportions)
    e = lambda p: p * np.log2(p + np.finfo(float).eps)
    return -e(proportions).sum()


def gini(s, attributes):
    """Return the attribute on which to split using Gini index."""
    gini_dict = dict()
    for attribute in attributes:
        gini_dict[attribute] = gini_impurity(s, attributes, attribute)
    return min(gini_dict, key=gini_dict.get)


def gini_impurity_helper(s, attribute, v):
    subset = s[s[attribute] == v]
    impurity = np.float64(1)
    for c in subset["label"].value_counts(normalize=True):
        impurity -= np.power(c, 2)
    return impurity * np.float64(len(subset) / len(s))


def gini_impurity(s: pd.DataFrame, _, attribute) -> np.floating:
    """Return the gini impurity of the subset s for the given attribute."""
    return sum(map(lambda v: gini_impurity_helper(s, attribute, v), s[attribute].unique()))
