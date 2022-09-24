from decision_tree import DecisionTree
from build_features import build_features
from split_functions import split_information_gain, gini, majority_error_split

if __name__ == "__main__":
    test_df, _, _ = build_features("car", True)
    for f in (split_information_gain, gini, majority_error_split):
        print(f"Testing {f.__name__}...")
        dt = DecisionTree(*build_features("car"), split_func=f)
        print(f"Prediction score: {round(dt.test_accuracy(test_df)*100, 4)}%.")
