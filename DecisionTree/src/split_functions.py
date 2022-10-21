"""split_functions.py contains methods for splitting the data in a decision tree"""
import math

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


def split_information_gain(s, attributes) -> str:
    """Return the attribute on which to split using information gain."""
    attr_dict = {attribute: information_gain(s, attributes, attribute) for attribute in attributes}
    return max(attr_dict, key=attr_dict.get)


def information_gain(s, attributes, attribute) -> float:
    """Return the information gain for a subset s for the given attribute."""
    gain = entropy(s)
    for v in attributes[attribute]:
        subset = s.loc[s[attribute] == v]
        gain -= (subset.size / s.size) * entropy(subset)
    return gain


def entropy(s, y="label") -> float:
    """Return the entropy of the subset s."""
    proportions = s[y].value_counts(normalize=True)
    return -sum(map(lambda n: n * math.log2(n), proportions))


def gini(s, attributes):
    """Return the attribute on which to split using Gini index."""
    gini_dict = dict()
    for attribute in attributes:
        gini_dict[attribute] = gini_impurity(s, attributes, attribute)
    return min(gini_dict, key=gini_dict.get)


def gini_impurity(s: pd.DataFrame, _, attribute) -> np.floating:
    """Return the gini impurity of the subset s for the given attribute."""
    return sum(map(lambda v: gini_impurity_helper(s, attribute, v), s[attribute].unique()))


def gini_impurity_helper(s, attribute, v):
    subset = s[s[attribute] == v]
    impurity = np.float64(1)
    for c in subset["label"].value_counts(normalize=True):
        impurity -= np.power(c, 2)
    return impurity * np.float64(len(subset) / len(s))


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "O": [
                "s",
                "s",
                "o",
                "r",
                "r",
                "r",
                "o",
                "s",
                "s",
                "r",
                "s",
                "o",
                "o",
                "r",
            ],
            "T": [
                "h",
                "h",
                "h",
                "m",
                "c",
                "c",
                "c",
                "m",
                "c",
                "m",
                "m",
                "m",
                "h",
                "m",
            ],
            "H": [
                "h",
                "h",
                "h",
                "h",
                "n",
                "n",
                "n",
                "h",
                "n",
                "n",
                "n",
                "h",
                "n",
                "h",
            ],
            "W": [
                "w",
                "s",
                "w",
                "w",
                "w",
                "s",
                "s",
                "w",
                "w",
                "w",
                "s",
                "s",
                "w",
                "s",
            ],
            "label": [
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
        }
    )
    attributes = {col: df[col].unique() for col in df}
    print(information_gain(df, attributes, "O"))
