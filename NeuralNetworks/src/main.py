import argparse

import numpy as np


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


def report_accuracy(w, X, y, test_X, test_y, predict=predict):
    print(f"{w=}")
    train_errors = sum([predict(X[i], w) != y[i] for i in range(y.size)])
    train_accuracy = (y.size - train_errors) / y.size
    print(f"{train_accuracy=}")
    test_errors = sum([predict(test_X[i], w) != test_y[i] for i in range(test_y.size)])
    test_accuracy = (test_y.size - test_errors) / test_y.size
    print(f"{test_accuracy=}\n")


def backprop(x, y, prediction, z, s, w):
    dw = np.zeros(w.shape)
    dz = np.zeros(z.shape)
    dy = prediction - y
    dw[2, :, 1] = dy * s[1, :]
    width = w.shape[2]
    dz[1] = dy * w[2, :, 0]
    for i in range(width):
        dw[1, :, i] = dz[1, i + 1] * sigmoid_d(z[1, i + 1]) * z[1, :]
    dz[0] = z[1, 1] * sigmoid_d(z[1, 0]) * w[2, :, 0] + z[1, 2] * sigmoid_d(z[1, 2]) * w[2, :, 1]
    for i in range(width):
        dw[0, :, i] = dz[0, i + 1] * sigmoid_d(z[0, i + 1]) * x[:]
    print(dw)


def forward_pass(x, input_w, hidden_w, output_w):
    z = np.concatenate((np.ones((2, 1)), np.zeros((2, hidden_w.shape[1]))), axis=1)
    s = np.concatenate((np.ones((2, 1)), np.zeros((2, hidden_w.shape[1]))), axis=1)
    for i in range(1, z.shape[1]):
        z[0, i] = x @ input_w[:, i - 1]
        s[0, i] = sigmoid(z[0, i])
    for i in range(1, z.shape[1]):
        z[1, i] = z[0] @ hidden_w[:, i - 1]
        s[1, i] = sigmoid(z[1, i])
    y = z[1] @ output_w
    return y, z, s


def q2a(layer_width=3):
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    # X, y = np.array([[1, 1, 1]]), np.array([1])
    num_layers = 3
    input_w = np.random.normal(size=(X.shape[1], layer_width))
    # input_w = np.array([[-1, 1], [-2, 2], [-3, 3]])
    hidden_w = np.array([[-1, 1], [-2, 2], [-3, 3]])
    output_w = np.array([-1, -2, -1.5])
    # w = np.reshape(
    # np.random.normal(size=num_layers * width * (width - 1)),
    # (num_layers, width, width - 1),
    # )

    # w = np.array(
    # [
    # [[-1, 1], [-2, 2], [-3, 3]],  # w^1
    # [[-1, 1], [-2, 2], [-3, 3]],  # w^2
    # [[-1, 0], [-2, 0], [-1.5, 0]],  # w^3
    # ]
    # )
    x = X[0]
    prediction, z, s = forward_pass(x, input_w, hidden_w, output_w)
    # backprop(x, y[0], prediction, z, s, w)


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
