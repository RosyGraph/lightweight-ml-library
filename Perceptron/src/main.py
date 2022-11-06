import argparse

import numpy as np


def sgn(x):
    return np.where(x > 0, +1, -1)


def parse_bank_note_csv(test=False):
    train_csv = np.loadtxt(f"data/bank-note/{'test' if test else 'train'}.csv", delimiter=",")
    _X, y = train_csv[:, :-1], sgn(train_csv[:, -1])
    m, _ = _X.shape
    X = np.concatenate((np.ones((m, 1)), _X), axis=1)
    return X, y


def standard_perceptron(X, y, epochs=1, r=0.5):
    w = np.array([np.zeros(X.shape[1])])
    for _ in range(epochs):
        rand_index = np.arange(y.shape[0])
        np.random.shuffle(rand_index)
        for i in rand_index:
            if X[i].dot(w[-1]) * y[i] <= 0:
                w = np.concatenate((w, np.array([w[-1] + r * (y[i] * X[i])])), axis=0)
    return w


def predict(x, w):
    return sgn(w.dot(x))


def test_accuracy(X, y, w):
    num_correct = 0
    m = y.shape[0]
    for i in range(m):
        if y[i] == predict(X[i], w):
            num_correct += 1
    return num_correct / m


def q2a():
    X, y = parse_bank_note_csv()
    w = standard_perceptron(X, y, epochs=10)[-1]
    accuracy = test_accuracy(*parse_bank_note_csv(test=True), w)
    print(f"{w=}")
    print(f"{accuracy=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Perceptron experiments.")
    parser.add_argument(
        "--q2a",
        action="store_true",
        help="Run the standard Perceptron on the bank-note data set for T = 1 to 10. Stores the learned weight vector and average prediction error in the reports/ directory.",
    )
    args = parser.parse_args()
    if args.q2a:
        q2a()
