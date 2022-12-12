import argparse

import numpy as np


def sgn(x):
    return np.where(x > 0, +1, -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid * (1 - sigmoid(x))


def loss(prediction, truth):
    return 0.5 * (prediction - truth) ** 2


def loss_gradient(prediction, truth):
    return prediction - truth


def parse_csv(train_path: str, test_path: str):
    train_csv = np.loadtxt(train_path, delimiter=",")
    _X, y = train_csv[:, :-1], sgn(train_csv[:, -1])
    m, _ = _X.shape
    X = np.concatenate((_X, np.ones((m, 1))), axis=1)
    test_csv = np.loadtxt(test_path, delimiter=",")
    _test_X, test_y = test_csv[:, :-1], sgn(test_csv[:, -1])
    test_m, _ = _test_X.shape
    test_X = np.concatenate((_test_X, np.ones((test_m, 1))), axis=1)
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


def problem2():
    w = np.array(
        [
            [[-1, 1], [-2, 2], [-3, 3]],  # w^1
            [[-1, 1], [-2, 2], [-3, 3]],  # w^2
            [[-1, 0], [-2, 0], [-1.5, 0]],  # w^3
        ]
    )

    x = np.ones(3)

    z11 = sigmoid(x @ w[0, :, 0])
    z12 = sigmoid(x @ w[0, :, 1])
    z1 = np.array([1, z11, z12])

    z21 = sigmoid(z1 @ w[1, :, 0])
    z22 = sigmoid(z1 @ w[1, :, 1])
    z2 = np.array([1, z21, z22])
    z = np.array([z1, z2])

    y = w[2, 0, 0] * z[1, 0] + w[2, 1, 0] * z[1, 1] + w[2, 2, 0] * z[1, 2]
    ystar = 1
    loss_y = ystar - y
    loss_w = np.zeros([3, 3, 2])
    loss_z = np.zeros([2, 3])
    loss_w[2, :, 1] = loss_y * z2
    loss_z[1, :] = w[2, :, 0] * loss_y
    s = w[1, 0, 0] * z[0, 0] + w[1, 1, 0] * z[0, 1] + w[1, 2, 0] * z[0, 2]
    loss_w[1, :, 0] = z[1, 1] * sigmoid_d(s) * z[0, :]
    s = w[1, 0, 1] * z[0, 0] + w[1, 1, 1] * z[0, 1] + w[1, 2, 1] * z[0, 2]
    loss_w[1, :, 1] = z[1, 2] * sigmoid_d(s) * z[0, :]

    s = w[1, 0, 0] * z[0, 0] + w[1, 1, 0] * z[0, 1] + w[1, 2, 0] * z[0, 2]
    z1 = sigmoid_d(s) * w[1, 0, 0]
    s = w[1, 0, 1] * z[0, 0] + w[1, 1, 1] * z[0, 1] + w[1, 2, 1] * z[0, 2]
    z2 = sigmoid_d(s) * w[1, 0, 1]
    loss_z[0, 0] = loss_z[1, 1] * z1 + loss_z[1, 2] * z2

    s = w[1, 0, 0] * z[0, 0] + w[1, 1, 0] * z[0, 1] + w[1, 2, 0] * z[0, 2]
    z1 = sigmoid_d(s) * w[1, 1, 0]
    s = w[1, 0, 1] * z[0, 0] + w[1, 1, 1] * z[0, 1] + w[1, 2, 1] * z[0, 2]
    z2 = sigmoid_d(s) * w[1, 1, 1]
    loss_z[0, 1] = loss_z[1, 1] * z1 + loss_z[1, 2] * z2

    s = w[1, 0, 0] * z[0, 0] + w[1, 1, 0] * z[0, 1] + w[1, 2, 0] * z[0, 2]
    z1 = sigmoid_d(s) * w[1, 1, 1]
    s = w[1, 0, 1] * z[0, 0] + w[1, 1, 1] * z[0, 1] + w[1, 2, 1] * z[0, 2]
    z2 = sigmoid_d(s) * w[1, 2, 1]
    loss_z[0, 2] = loss_z[1, 1] * z1 + loss_z[1, 2] * z2

    s = x[0] * w[0, 0, 0] + x[1] * w[0, 1, 0] + x[2] * w[0, 2, 0]
    loss_w[0, 0, 0] = loss_z[0, 1] * sigmoid_d(s) * x[0]
    loss_w[0, 1, 0] = loss_z[0, 1] * sigmoid_d(s) * x[1]
    loss_w[0, 2, 0] = loss_z[0, 1] * sigmoid_d(s) * x[2]
    s = x[0] * w[0, 0, 1] + x[1] * w[0, 1, 1] + x[2] * w[0, 2, 1]
    loss_w[0, 0, 1] = loss_z[0, 2] * sigmoid_d(s) * x[0]
    loss_w[0, 1, 1] = loss_z[0, 2] * sigmoid_d(s) * x[1]
    loss_w[0, 2, 1] = loss_z[0, 2] * sigmoid_d(s) * x[2]


def backprop():
    pass


def forward_pass(x, w):
    z = np.zeros((w.shape[0] - 1, w.shape[1]))
    s = np.zeros((w.shape[0] - 1, w.shape[1]))
    temp = x
    for j, layer in enumerate(w[:-1]):
        z[j, :] = np.array([1] + [temp @ layer[:, i] for i in range(w.shape[2])])
        temp = np.append(z[j, 0], sigmoid(z[j, 1:]))
        s[j, :] = temp
    y = temp @ w[2, :, 0]
    return y, z, s


def q2a(layer_width=3):
    X, y, test_X, test_y = parse_csv("./data/bank-note/train.csv", "./data/bank-note/test.csv")
    num_layers = 3
    # w = np.reshape(
    # np.random.normal(size=num_layers * layer_width * (layer_width - 1)),
    # (num_layers, layer_width, layer_width - 1),
    # )

    w = np.array(
        [
            [[-1, 1], [-2, 2], [-3, 3]],  # w^1
            [[-1, 1], [-2, 2], [-3, 3]],  # w^2
            [[-1, 0], [-2, 0], [-1.5, 0]],  # w^3
        ]
    )
    x = np.ones(3)
    prediction = forward_pass(x, w)
    backprop()
    print(prediction)


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
