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


def report_ssgd_accuracy(w, X, y, test_X, test_y):
    print(f"{w=}")
    train_errors = sum([predict(X[i], w) != y[i] for i in range(y.size)])
    train_accuracy = (y.size - train_errors) / y.size
    print(f"{train_accuracy=}")
    test_errors = sum([predict(test_X[i], w) != test_y[i] for i in range(test_y.size)])
    test_accuracy = (test_y.size - test_errors) / test_y.size
    print(f"{test_accuracy=}\n")


def q2a():
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    hyperparameters = np.array([100 / 873, 500 / 873, 700 / 873])
    w = np.zeros(X.shape[1])
    r0 = 0.015
    a = 0.01
    schedule = lambda t: r0 / (1 + (r0 / a) * t)
    rng = np.random.default_rng()
    for C in hyperparameters:
        w = np.zeros(X.shape[1])
        print(f"{C=}")
        for t in range(1, 101):
            r = schedule(t)
            for _ in range(y.size):
                i = rng.integers(0, y.size)
                if y[i] * w @ X[i] <= 1:
                    w = w - r * np.append(w[:-1], 0) + r * C * y.size * y[i] * X[i]
                else:
                    w = np.append((1 - r) * w[:-1], w[-1])
        report_ssgd_accuracy(w, X, y, test_X, test_y)
    return w


def q2b():
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    hyperparameters = np.array([100 / 873, 500 / 873, 700 / 873])
    w = np.zeros(X.shape[1])
    r0 = 0.011
    schedule = lambda t: r0 / (1 + t)
    rng = np.random.default_rng()
    for C in hyperparameters:
        w = np.zeros(X.shape[1])
        print(f"{C=}")
        for t in range(1, 101):
            r = schedule(t)
            for _ in range(y.size):
                i = rng.integers(0, y.size)
                if y[i] * w @ X[i] <= 1:
                    w = w - r * np.append(w[:-1], 0) + r * C * y.size * y[i] * X[i]
                else:
                    w = np.append((1 - r) * w[:-1], w[-1])
        report_ssgd_accuracy(w, X, y, test_X, test_y)
    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for SVM experiments.")
    assign_group = parser.add_argument_group("programming practice experiments")
    assign_group.add_argument(
        "--q2a",
        action="store_true",
        help="report learned weights and accuracy for SSGD for various C with schedule r0/(1+(r0/a)t)",
    )
    assign_group.add_argument(
        "--q2b",
        action="store_true",
        help="report learned weights and accuracy for SSGD for various C with schedule r0/(1+t)",
    )
    assign_group.add_argument(
        "--all",
        action="store_true",
        help="run each experiment listed above",
    )
    args = parser.parse_args()
    if args.all:
        print("Question 2 part (a)...")
        q2a()
        print("Question 2 part (b)...")
        q2b()
    elif args.q2a:
        q2a()
    elif args.q2b:
        q2b()
