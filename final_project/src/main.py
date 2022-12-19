from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


def build_income() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(f"data/processed/train_final.csv")
    df = df.astype({"income>50K": "bool"})
    df = df.rename(columns={"income>50K": "Prediction"})
    for c, t in zip(df.columns, df.dtypes):
        if t == "object":
            df[c] = df[c].astype("category")
    arr = np.stack([df[c].cat.codes if t == "category" else df[c] for c, t in zip(df.columns, df.dtypes)], 1)
    X, y = arr[:, :-1], arr[:, -1]
    return X, y


def build_income_test():
    df = pd.read_csv(f"data/processed/test_final.csv")
    for c, t in zip(df.columns, df.dtypes):
        if t == "object":
            df[c] = df[c].astype("category")
    return np.stack([df[c].cat.codes if t == "category" else df[c] for c, t in zip(df.columns, df.dtypes)], 1)


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class Net(nn.Module):
    def __init__(self, input_shape, shapes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, shapes[0])
        self.fc2 = nn.Linear(shapes[0], shapes[1])
        self.fc3 = nn.Linear(shapes[1], 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


if __name__ == "__main__":
    X, y = build_income()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    trainset = dataset(X, y)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    shape_permutations = [[16, 16], [16, 32], [16, 64], [32, 16], [32, 32], [32, 64], [64, 16], [64, 32], [64, 64]]
    loss_funcs = {"L1Loss": nn.L1Loss(), "MSELoss": nn.MSELoss()}
    for loss_func_str, loss_func in loss_funcs.items():
        for lri, lr in enumerate({0.5, 0.1, 0.05, 0.01}):
            for p, shapes in enumerate(shape_permutations):
                epochs = 500
                model = Net(input_shape=X.shape[1], shapes=shapes)
                momentum = 0.9
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
                loss_fn = loss_func
                accuracy = []
                for i in range(epochs):
                    acc = 0
                    for (x_train, y_train) in trainloader:
                        output = model(x_train)
                        loss = loss_fn(output, y_train.reshape(-1, 1))
                        predicted = model(torch.tensor(X, dtype=torch.float32))
                        acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    accuracy.append(acc)
                test_X = torch.Tensor(scaler.fit_transform(build_income_test()[:, 1:]))

                with torch.no_grad():
                    model.eval()
                    results = ["ID,Prediction"] + [
                        ",".join(map(str, (i + 1, model(x).round().int().item()))) for i, x in enumerate(test_X)
                    ]

                parent = Path("reports") / Path(f"nn-{loss_func_str}-{lri}-{p}")
                parent.mkdir(exist_ok=True)
                with open(parent / Path("results.csv"), "w") as f:
                    f.write("\n".join(results))
                s = pd.DataFrame(accuracy, columns=["acc"])
                s.to_csv(parent / Path("accuracy.csv"))
                with open(parent / Path("desc.txt"), "w") as f:
                    f.write(f"{shapes=}\n{lr=}\n{epochs=}\n{momentum=}\nloss_func={loss_func_str}")
