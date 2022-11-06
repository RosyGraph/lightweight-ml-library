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
        rand_index = np.arange(y.size)
        np.random.shuffle(rand_index)
        for i in rand_index:
            if X[i].dot(w[-1]) * y[i] <= 0:
                w = np.concatenate((w, np.array([w[-1] + r * (y[i] * X[i])])), axis=0)
    return w


def voted_perceptron(X, y, epochs=1, r=0.5):
    w = np.array([np.zeros(X.shape[1])])
    C = np.zeros(1)
    for _ in range(epochs):
        for i in range(y.size):
            if X[i].dot(w[-1]) * y[i] <= 0:
                w = np.concatenate((w, np.array([w[-1] + r * (y[i] * X[i])])), axis=0)
                C = np.concatenate((C, np.ones(1)), axis=0)
            else:
                C[-1] += 1
    return w, C


def predict(x, w):
    return sgn(w.dot(x))


def weighted_predict(x, w, C):
    return sgn(sum([C[i] * sgn(w[i].dot(x)) for i in range(C.size)]))


def test_accuracy(X, y, prediction_fn):
    num_correct = 0
    m = y.size
    for i in range(m):
        if y[i] == prediction_fn(X[i]):
            num_correct += 1
    return num_correct / m


def q2a():
    X, y = parse_bank_note_csv()
    w = standard_perceptron(X, y, epochs=10)[-1]
    accuracy = test_accuracy(*parse_bank_note_csv(test=True), lambda x: predict(x, w))
    print("Question 2 part (a): standard Perceptron")
    print("Learned weight vector")
    print(np.array2string(w))
    print()
    print(f"Accuracy over the test data set: {accuracy}")


def q2b():
    X, y = parse_bank_note_csv()
    w, C = voted_perceptron(X, y, epochs=10)
    accuracy = test_accuracy(*parse_bank_note_csv(test=True), lambda x: weighted_predict(x, w, C))
    print("Question 2 part (b): voted Perceptron")
    print("Learned weight vector")
    print(np.array2string(w))
    print()
    print("Number of correctly predicted training examples")
    print(np.array2string(C))
    print()
    print(f"Accuracy over the test data set: {accuracy}")


def q2c():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Perceptron experiments.")
    parser.add_argument(
        "--q2a",
        action="store_true",
        help="run the standard Perceptron on the bank-note data set for T=10",
    )
    parser.add_argument(
        "--q2b",
        action="store_true",
        help="run the voted Perceptron on the bank-note data set for T=10",
    )
    parser.add_argument(
        "--q2c",
        action="store_true",
        help="run the average Perceptron on the bank-note data set for T=10",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run standard, voted, and average Perceptron on the bank-note data set for T=10",
    )
    args = parser.parse_args()
    if args.all:
        q2a()
        print()
        q2b()
        print()
        q2c()
    if args.q2a:
        q2a()
    if args.q2b:
        q2b()
    if args.q2c:
        q2c()
