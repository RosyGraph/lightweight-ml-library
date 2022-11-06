import argparse

import numpy as np


def sgn(x):
    return np.where(x > 0, +1, -1)


def parse_bank_note_csv():
    train_csv = np.loadtxt("data/bank-note/train.csv", delimiter=",")
    _X, y = train_csv[:, :-1], sgn(train_csv[:, -1])
    m, _ = _X.shape
    X = np.concatenate((np.ones((m, 1)), _X), axis=1)
    return X, y


def q2a():
    X, y = parse_bank_note_csv()
    w = np.zeros(X.shape[1])
    print(w)


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
