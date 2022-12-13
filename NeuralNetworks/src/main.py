import argparse
import random

import numpy as np

NUM_LAYERS = 3


def sgn(x):
    return np.where(x > 0, +1, -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def loss(prediction, truth):
    return 0.5 * (prediction - truth) ** 2


def loss_gradient(prediction, truth):
    return prediction - truth


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
    X = np.concatenate((_X, np.ones((m, 1))), axis=1)
    return X, y


def predict(x, w):
    return sgn(w.dot(x))


def practice_setup():
    X, y = np.array([[1, 1, 1]]), np.array([1])
    input_w = np.array([[-1, 1], [-2, 2], [-3, 3]])
    hidden_w = np.array([[-1, 1], [-2, 2], [-3, 3]])
    output_w = np.array([-1, -2, -1.5])
    return X, y, input_w, hidden_w, output_w


def bank_setup(layer_width=3):
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    input_w = np.random.normal(size=(X.shape[1], layer_width - 1))
    hidden_w = np.random.normal(size=(layer_width, layer_width - 1))
    output_w = np.random.normal(size=layer_width)
    return X, y, test_X, test_y, input_w, hidden_w, output_w


def report_accuracy(X, y, test_X, test_y, input_w, hidden_w, output_w):
    train_errors = 0
    for i in range(y.size):
        prediction, *_ = forward_pass(X[i], input_w, hidden_w, output_w)
        if sgn(prediction) != y[i]:
            train_errors += 1
    train_accuracy = (y.size - train_errors) / y.size
    print(f"{train_accuracy=}")
    test_errors = 0
    for i in range(test_y.size):
        prediction, *_ = forward_pass(test_X[i], input_w, hidden_w, output_w)
        if sgn(prediction) != test_y[i]:
            test_errors += 1
    test_accuracy = (test_y.size - test_errors) / test_y.size
    print(f"{test_accuracy=}")


def backprop(x, y, prediction, z, s, input_w, hidden_w, output_w):
    d_input_w = np.zeros(input_w.shape)
    d_hidden_w = np.zeros(hidden_w.shape)
    d_output_w = np.zeros(output_w.shape)
    dz = np.zeros(z.shape)
    dy = prediction - y
    width = hidden_w.shape[1]
    d_output_w = dy * s[1]
    dz[1] = dy * output_w
    for i in range(width):
        d_hidden_w[:, i] = dz[1, i + 1] * sigmoid_d(z[1, i + 1]) * z[1, :]
    for i in range(dz.shape[1]):
        dz[0, i] = np.array([dz[1, j + 1] * sigmoid_d(z[1, j + 1]) * hidden_w[i, j] for j in range(width)]).sum()
    for i in range(width):
        d_input_w[:, i] = dz[0, i + 1] * sigmoid_d(z[0, i + 1]) * x
    return d_input_w, d_hidden_w, d_output_w, dz


def forward_pass(x, input_w, hidden_w, output_w):
    z = np.concatenate((np.zeros((2, 1)), np.zeros((2, hidden_w.shape[1]))), axis=1)
    s = np.concatenate((np.zeros((2, 1)), np.zeros((2, hidden_w.shape[1]))), axis=1)
    for i in range(1, z.shape[1]):
        z[0, i] = x @ input_w[:, i - 1]
        s[0, i] = sigmoid(z[0, i])
    for i in range(1, z.shape[1]):
        z[1, i] = s[0] @ hidden_w[:, i - 1]
        s[1, i] = sigmoid(z[1, i])
    y = s[1] @ output_w
    return y, z, s


def q2a(layer_width=3):
    X, y, _, _, input_w, hidden_w, output_w = bank_setup(layer_width)
    # X, y, input_w, hidden_w, output_w = practice_setup()
    x = X[0]
    prediction, z, s = forward_pass(x, input_w, hidden_w, output_w)
    d_input_w, d_hidden_w, d_output_w, dz = backprop(x, y[0], prediction, z, s, input_w, hidden_w, output_w)
    print(f"Neurons:\n{dz}")
    print(f"Input layer:\n{d_input_w}")
    print(f"Hidden layer:\n{d_hidden_w}")
    print(f"Output layer:\n{d_output_w}")


def q2b():
    r0 = 0.02
    d = 1.5
    schedule = lambda t: r0 / (1 + (r0 / d) * t)
    for layer_width in [5, 10, 25, 50, 100]:
        print(f"Width of hidden layers: {layer_width}...")
        X, y, test_X, test_y, input_w, hidden_w, output_w = bank_setup(layer_width)
        for t in range(1, 500):
            i = random.randrange(0, y.size)
            x, yi = X[i], y[i]
            prediction, z, s = forward_pass(x, input_w, hidden_w, output_w)
            d_input_w, d_hidden_w, d_output_w, dz = backprop(x, yi, prediction, z, s, input_w, hidden_w, output_w)
            input_w = input_w - schedule(t) * d_input_w
            hidden_w = hidden_w - schedule(t) * d_hidden_w
            output_w = output_w - schedule(t) * d_output_w
        report_accuracy(X, y, test_X, test_y, input_w, hidden_w, output_w)


def q2c():
    r0 = 0.07
    d = 1.5
    schedule = lambda t: r0 / (1 + (r0 / d) * t)
    for layer_width in [5, 10, 25, 50, 100]:
        print(f"Width of hidden layers: {layer_width}...")
        X, y, test_X, test_y, input_w, hidden_w, output_w = bank_setup(layer_width)
        input_w = np.zeros(shape=input_w.shape)
        hidden_w = np.zeros(shape=hidden_w.shape)
        output_w = np.zeros(shape=output_w.shape)
        for t in range(1, 500):
            i = random.randrange(0, y.size)
            x, yi = X[i], y[i]
            prediction, z, s = forward_pass(x, input_w, hidden_w, output_w)
            d_input_w, d_hidden_w, d_output_w, dz = backprop(x, yi, prediction, z, s, input_w, hidden_w, output_w)
            input_w = input_w - schedule(t) * d_input_w
            hidden_w = hidden_w - schedule(t) * d_hidden_w
            output_w = output_w - schedule(t) * d_output_w
        report_accuracy(X, y, test_X, test_y, input_w, hidden_w, output_w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for neural network experiments")
    assign_group = parser.add_argument_group("programming practice experiments")
    assign_group.add_argument(
        "--q2a",
        action="store_true",
        help="compute the gradient with respect to all edge weights given the first training example",
    )
    assign_group.add_argument(
        "--q2b",
        action="store_true",
        help="run stochastic gradient descent on bank data and report training and test error for various layer widths with weights initialized to random values from the standard normal distribution",
    )
    assign_group.add_argument(
        "--q2c",
        action="store_true",
        help="run stochastic gradient descent on bank data and report training and test error for various layer widths with weights initialized to zeros",
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
        print("Question 2 part (b)...")
        q2b()
        print("Question 2 part (c)...")
        q2c()
    elif args.q2a:
        q2a()
    elif args.q2b:
        q2b()
    elif args.q2c:
        q2c()
