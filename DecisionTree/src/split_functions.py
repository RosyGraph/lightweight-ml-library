import math


def majority_error_split(s, attributes, label="label"):
    me_dict = dict()
    me = majority_error(s, label)
    for attribute in attributes:
        tmp = me
        for value, p in zip(s[attribute].value_counts(normalize=True).index, s[attribute].value_counts(normalize=True)):
            subset = s[s[attribute] == value]
            tmp -= p * majority_error(subset, label)
        me_dict[attribute] = tmp
    return max(me_dict, key=me_dict.get)


def majority_error(s, label="label"):
    return 1 - s[label].value_counts(normalize=True)[0]


def information_gain(s, attributes, attribute):
    gain = entropy(s)
    for v in attributes[attribute]:
        subset = s.loc[s[attribute] == v]
        gain -= (subset.size / s.size) * entropy(s)
    return gain


def entropy(s, y="label"):
    proportions = s[y].value_counts(normalize=True)
    return -sum(map(lambda n: n * math.log2(n), proportions))


def split_information_gain(s, attributes):
    attr_dict = {attribute: information_gain(s, attributes, attribute) for attribute in attributes}
    return max(attr_dict, key=attr_dict.get)


def gini_impurity(s, attributes, attribute):
    leaf_impurities = []
    for v in s[attribute].unique():
        subset = s[s[attribute] == v]
        impurity = 1
        for c in subset["label"].value_counts(normalize=True):
            impurity -= c**2
        wt = len(subset) / len(s)
        leaf_impurities.append(wt * impurity)
    return sum(leaf_impurities)


def gini(s, attributes, label="label"):
    gini_dict = dict()
    for attribute in attributes:
        gini_dict[attribute] = gini_impurity(s, attributes, attribute)
    return min(gini_dict, key=gini_dict.get)
