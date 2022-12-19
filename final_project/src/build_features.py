"""build_features.py contains methods for extracting the features from datasets"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def build_income_df(test=False) -> Tuple[pd.DataFrame, Optional[dict], Optional[set]]:
    attributes = dict()
    if test:
        df = pd.read_csv(f"data/processed/test_final.csv")
    else:
        df = pd.read_csv(f"data/processed/train_final.csv")
        df = df.astype({"income>50K": "bool"})
        df = df.rename(columns={"income>50K": "Prediction"})
    for numeric_column_name in ("education.num", "hours.per.week", "fnlwgt", "age"):
        df[numeric_column_name] = pd.qcut(df[numeric_column_name], q=2, duplicates="drop")
    for numeric_column_name in ("capital.gain", "capital.loss"):
        df[numeric_column_name] = df[numeric_column_name].astype("bool")
    for column in df.columns:
        mode = df[column].value_counts()
        new = mode.index[0]
        if new == "?":
            new = mode.index[1]
        df[column].replace("?", new, inplace=True)
    if test:
        return df, None, None
    attribute_keys = set(df.columns) - {"Prediction", "ID"}
    attributes = dict(zip(attribute_keys, map(lambda c: list(df[c].unique()), attribute_keys)))
    return (df, attributes, set(df["Prediction"].unique()))


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


if __name__ == "__main__":
    X, y = build_income()
