import argparse

import numpy as np


def sgn(x):
    return np.where(x > 0, +1, -1)


def parse_csv(train_path: str, test_path: str):
    train_csv = np.loadtxt(train_path, delimiter=",")
    _X, y = train_csv[:, :-1], sgn(train_csv[:, -1])
    m, _ = _X.shape
    X = np.concatenate((np.ones((m, 1)), _X), axis=1)
    test_csv = np.loadtxt(test_path, delimiter=",")
    _test_X, test_y = test_csv[:, :-1], sgn(test_csv[:, -1])
    test_m, _ = _test_X.shape
    test_X = np.concatenate((np.ones((test_m, 1)), _test_X), axis=1)
    return X, y, test_X, test_y


def parse_bank_note_csv(test=False):
    train_csv = np.loadtxt(f"data/bank-note/{'test' if test else 'train'}.csv", delimiter=",")
    _X, y = train_csv[:, :-1], sgn(train_csv[:, -1])
    m, _ = _X.shape
    X = np.concatenate((np.ones((m, 1)), _X), axis=1)
    return X, y


def predict(x, w):
    return sgn(w.dot(x))


def report_accuracy(w, X, y, test_X, test_y, predict=predict):
    print(f"{w=}")
    train_errors = sum([predict(X[i], w) != y[i] for i in range(y.size)])
    train_accuracy = (y.size - train_errors) / y.size
    print(f"{train_accuracy=}")
    test_errors = sum([predict(test_X[i], w) != test_y[i] for i in range(test_y.size)])
    test_accuracy = (test_y.size - test_errors) / test_y.size
    print(f"{test_accuracy=}\n")


def q2a():
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    print("Hello machine learning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for neural network experiments")
    assign_group = parser.add_argument_group("programming practice experiments")
    assign_group.add_argument(
        "--q2a",
        action="store_true",
        help="compute the gradient with respect to all edge weights given the first training example",
    )
    assign_group.add_argument(
        "--all",
        action="store_true",
        help="run all of the above experiments (this may take awhile)",
    )
    args = parser.parse_args()
    if args.all:
        print("Question 2 part (a)...")
        q2a()
    elif args.q2a:
        q2a()
