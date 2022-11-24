import argparse

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


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


def vectorized_gaussian_kernel(X, g):
    """
    Credit to stackexchange.com user bayerj
    https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
    """
    return np.exp(-squareform(pdist(X, "euclidean")) ** 2 / g)


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
        report_accuracy(w, X, y, test_X, test_y)
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
        report_accuracy(w, X, y, test_X, test_y)
    return w


def q3a():
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    hyperparameters = np.array([100 / 873, 500 / 873, 700 / 873])
    A = np.multiply(X @ X.T, y.reshape(y.size, 1) @ y.reshape(1, y.size))
    objective = lambda a: 0.5 * (a.T @ A @ a) - a.sum()
    for C in hyperparameters:
        print(f"{C=}")
        bounds = [(0, C) for _ in range(y.size)]
        starting_guess = np.zeros(y.size)
        constraints = {"type": "eq", "fun": lambda a: a @ y}
        a = minimize(objective, starting_guess, method="SLSQP", bounds=bounds, constraints=[constraints]).x
        support_vectors = np.array([X[i] for i in range(a.size) if a[i] > 0])
        w = np.array([a[i] * y[i] * X[i] for i in range(y.size)]).sum(axis=0)
        b = np.array([y[j] - w.T @ X[j] for j in range(y.size)]).sum() / y.size
        print(f"{w=}")
        print(f"{b=}")
        print(f"{support_vectors=}")
        report_accuracy(w, X, y, test_X, test_y, predict=lambda x, w: sgn(w.T @ x + b))


def q3b():
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    hyperparameters = np.array([100 / 873, 500 / 873, 700 / 873])
    gamma = np.array([0.1, 0.5, 1, 5, 100])
    support_vectors = np.array([])
    for C in hyperparameters:
        print(f"{C=}")
        for g in gamma:
            A = np.multiply(vectorized_gaussian_kernel(X, g), y.reshape(y.size, 1) @ y.reshape(1, y.size))
            objective = lambda a: 0.5 * (a.T @ A @ a) - a.sum()
            print(f"gamma={g}")
            bounds = [(0, C) for _ in range(y.size)]
            starting_guess = np.zeros(y.size)
            constraints = {"type": "eq", "fun": lambda a: a @ y}
            a = minimize(objective, starting_guess, method="SLSQP", bounds=bounds, constraints=[constraints]).x
            phi = lambda xi, xj: np.exp(-(np.linalg.norm(xi - xj) ** 2) / g)
            b = (
                np.array(
                    [y[j] - np.array([a[i] * y[i] * phi(X[i], X[j]) for i in range(y.size)]) for j in range(y.size)]
                ).sum()
                / y.size
            )
            print(f"{b=}")
            print(f"num support vectors={np.array([X[i] for i in range(a.size) if a[i] > 0]).shape}")
            if C == 500 / 873:
                support_vectors = np.append(support_vectors, np.array([X[i] for i in range(a.size) if a[i] > 0]))

            def predict(x):
                return sgn(np.array([a[i] * y[i] * phi(X[i], x) for i in range(y.size)]).sum())

            train_errors = sum([predict(X[i]) != y[i] for i in range(y.size)])
            train_accuracy = (y.size - train_errors) / y.size
            print(f"{train_accuracy=}")
            test_errors = sum([predict(test_X[i]) != test_y[i] for i in range(test_y.size)])
            test_accuracy = (test_y.size - test_errors) / test_y.size
            print(f"{test_accuracy=}\n")


def q3c():
    X, y, _, _ = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    gamma = np.array([0.1, 0.5, 1, 5, 100])
    support_vectors = np.array([])
    C = 500 / 873
    for g in gamma:
        A = np.multiply(vectorized_gaussian_kernel(X, g), y.reshape(y.size, 1) @ y.reshape(1, y.size))
        objective = lambda a: 0.5 * (a.T @ A @ a) - a.sum()
        print(f"gamma={g}")
        bounds = [(0, C) for _ in range(y.size)]
        starting_guess = np.zeros(y.size)
        constraints = {"type": "eq", "fun": lambda a: a @ y}
        a = minimize(objective, starting_guess, method="SLSQP", bounds=bounds, constraints=[constraints]).x
        overlap = 0
        for v in support_vectors:
            for row in np.array([X[i] for i in range(a.size) if a[i] > 0]):
                overlap += row.tolist() in v.tolist()
        print(f"{overlap=}")
        support_vectors = np.array([X[i] for i in range(a.size) if a[i] > 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for SVM experiments")
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
        "--q3a",
        action="store_true",
        help="report learned weights and accuracy for dual SVM for various C",
    )
    assign_group.add_argument(
        "--q3b",
        action="store_true",
        help="report support vectors and accuracy for dual SVM for various C and gamma",
    )
    assign_group.add_argument(
        "--q3c",
        action="store_true",
        help="report overlapping support vectors for various gamma",
    )
    assign_group.add_argument(
        "--all",
        action="store_true",
        help="run each experiment listed above (this may take awhile)",
    )
    args = parser.parse_args()
    if args.all:
        print("Question 2 part (a)...")
        q2a()
        print("Question 2 part (b)...")
        q2b()
        print("Question 3 part (a)...")
        q3a()
        print("Question 3 part (b)...")
        q3b()
        print("Question 3 part (b)...")
        q3c()
    elif args.q2a:
        q2a()
    elif args.q2b:
        q2b()
    elif args.q3a:
        q3a()
    elif args.q3b:
        q3b()
    elif args.q3c:
        q3c()
