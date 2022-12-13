import argparse

import numpy as np
from scipy import special


def logistic_grad(X, y, w, v):
    new_w = np.copy(w)
    for i in range(X.shape[1]):
        new_w[i] = 2 * w[i] * v - np.sum(
            [
                X[j, i] * y[j] * special.expit(-y[j] * w @ X[j, :]) / (1 + special.expit(-y[j] * w @ X[j, :]))
                for j in np.random.permutation(X.shape[0])
            ]
        )
    return new_w


def sgn(x):
    return np.where(x > 0, +1, -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def report_accuracy(X, y, test_X, test_y, w):
    train_errors = 0
    for i in range(y.size):
        if sgn(w @ X[i]) != y[i]:
            train_errors += 1
    train_accuracy = (y.size - train_errors) / y.size
    print(f"{train_accuracy=}")
    test_errors = 0
    for i in range(test_y.size):
        if sgn(w @ test_X[i]) != test_y[i]:
            test_errors += 1
    train_accuracy = (y.size - train_errors) / y.size
    test_accuracy = (test_y.size - test_errors) / test_y.size
    print(f"{test_accuracy=}")


def q3a():
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    w = np.zeros(shape=X.shape[1])
    r0 = 0.05
    d = 1.5
    schedule = lambda t: r0 / (1 + (r0 / d) * t)
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    for v in variances:
        print(f"Variance: {v}...")
        for t in range(100):
            rate = schedule(t)
            w = w - rate * logistic_grad(X, y, w, v)
        report_accuracy(X, y, test_X, test_y, w)


def q3b():
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    n = y.size
    w = np.random.normal(size=X.shape[1])
    r0 = 0.01
    d = 1.5
    schedule = lambda t: r0 / (1 + (r0 / d) * t)
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    for v in variances:
        print(f"Variance: {v}...")
        for t in range(100):
            rate = schedule(t)
            w = (
                w
                - rate * (1 / n) * np.array([(y[i] - sigmoid(w @ X[i])) * X[i] for i in np.random.permutation(n)]).sum()
            )
        report_accuracy(X, y, test_X, test_y, w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for logistic regression experiments")
    assign_group = parser.add_argument_group("programming practice experiments")
    assign_group.add_argument(
        "--q3a",
        action="store_true",
        help="report training and test accuracy for SGD for MAP estimation for different variances",
    )
    assign_group.add_argument(
        "--q3b",
        action="store_true",
        help="report training and test accuracy for SGD for ML estimation for different variances",
    )
    assign_group.add_argument(
        "--all",
        action="store_true",
        help="run all of the above experiments",
    )
    args = parser.parse_args()
    if args.all:
        print("Question 3 part (a)...")
        q3a()
        print("Question 3 part (b)...")
        q3b()
    elif args.q3a:
        q3a()
    elif args.q3b:
        q3b()
