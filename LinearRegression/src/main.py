import random

import numpy as np


def q4a(train_data, test_data):
    xs, y = train_data[:, 0:-1], train_data[:, -1]
    x1 = np.ones(xs.shape[0]).reshape(xs.shape[0], 1)
    xs = np.concatenate((x1, xs), axis=1)
    m, d = xs.shape
    r = 0.01
    t = 0
    runs = 10_000
    converged = False
    w = np.zeros(d).reshape(1, d)
    results_table = []
    while not converged:
        w = np.zeros(d).reshape(1, d)
        gradient = np.zeros(w.shape)
        diff = np.finfo(float).max
        results_table = [["t", "cost"] + list(map(str, (f"x{i}" for i in range(d))))]
        while diff > 10e-6 and t < runs:
            for j in range(d):
                gradient[0][j] = -np.array(
                    [(y[i] - np.dot(w, xs[i].reshape(1, d).T).item()) * xs[i][j] for i in range(m)]
                ).sum()
            cost = 0.5 * np.array([(y[i] - np.dot(w, xs[i].reshape(1, d).T).item()) for i in range(m)])
            results_table.append([str(t)] + list(map(str, (c for c in cost))))
            old_w = w
            step = r * gradient
            w = w - step
            diff = np.linalg.norm(old_w - w)
            t += 1
        if t == runs or np.isnan(np.sum(w)):
            r /= 2
            break
        else:
            converged = True
    with open("reports/q4a.csv", "w+") as f:
        for row in results_table:
            f.write(",".join(row) + "\n")
    test_xs, test_y = test_data[:, 0:-1], test_data[:, -1]
    test_x1 = np.ones(test_xs.shape[0]).reshape(test_xs.shape[0], 1)
    test_xs = np.concatenate((test_x1, test_xs), axis=1)
    m, d = test_xs.shape
    test_cost = 0.5 * np.array([(test_y[i] - np.dot(w, test_xs[i].reshape(1, d).T).item()) for i in range(m)])
    with open("reports/q4a_result.txt", "w+") as f:
        f.write(f"{r=}, {w[0]=}, {test_cost=}")


def q4b(train_data, test_data):
    xs, y = train_data[:, 0:-1], train_data[:, -1]
    x1 = np.ones(xs.shape[0]).reshape(xs.shape[0], 1)
    xs = np.concatenate((x1, xs), axis=1)
    m, d = xs.shape
    r = 0.01
    t = 0
    runs = 500
    w = np.zeros(d).reshape(1, d)
    results_table = [
        ["t"] + [f"c{i}" for i in range(m)],
    ]
    for t in range(runs):
        i = random.randrange(m)
        for j in range(d):
            w[0][j] = w[0][j] + r * (y[i] - np.dot(w, train_data[i].reshape(1, d).T).item()) * train_data[i][j]
        cost = 0.5 * np.array([(y[i] - np.dot(w, xs[i].reshape(1, d).T).item()) for i in range(m)])
        results_table.append([str(t)] + list(map(str, cost)))
    with open("reports/q4b.csv", "w+") as f:
        for row in results_table:
            f.write(",".join(row) + "\n")
    test_xs, test_y = test_data[:, 0:-1], test_data[:, -1]
    test_x1 = np.ones(test_xs.shape[0]).reshape(test_xs.shape[0], 1)
    test_xs = np.concatenate((test_x1, test_xs), axis=1)
    m, d = test_xs.shape
    test_cost = 0.5 * np.array([(test_y[i] - np.dot(w, test_xs[i].reshape(1, d).T).item()) for i in range(m)])
    with open("reports/q4b_results.txt", "w+") as f:
        f.write("w\n")
        f.write(str(w) + "\n")
        f.write("test_cost\n")
        f.write(str(test_cost) + "\n")


def q4c(train_data):
    xs, y = train_data[:, 0:-1], train_data[:, -1]
    x1 = np.ones(xs.shape[0]).reshape(xs.shape[0], 1)
    xs = np.concatenate((x1, xs), axis=1)
    a = xs.T.dot(xs)
    b = xs.T.dot(y)
    print(np.linalg.solve(a, b))


if __name__ == "__main__":
    train_data = np.genfromtxt("data/concrete/train.csv", delimiter=",")
    test_data = np.genfromtxt("data/concrete/test.csv", delimiter=",")
    q4a(train_data, test_data)
    q4b(train_data, test_data)
    q4c(train_data)
