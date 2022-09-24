"""build_features.py contains methods for extracting the features from datasets"""
import os
import pandas as pd
import numpy as np
from typing import Tuple


def build_features(dataset, test=False) -> Tuple[pd.DataFrame, dict, set]:
    if dataset == "car":
        return build_car_features(test)
    if dataset == "bank":
        return build_bank_features(test)
    raise Exception(f"'{dataset}' is not a known dataset.")


def build_car_features(test=False) -> Tuple[pd.DataFrame, dict, set]:
    """Build the features of the given dataset and load the data into a Pandas dataframe."""
    path_to_data_desc = os.path.join("data", "car", "data-desc.txt")
    attributes = dict()
    labels = set()
    columns = []
    with open(path_to_data_desc, "r") as f:
        parse_attributes = False
        parse_columns = False
        for line in f.readlines():
            if parse_columns:
                columns = list(map(lambda s: s.strip(), line.split(",")))
            if "| attributes" in line:
                parse_attributes = True
                continue
            if parse_attributes:
                attribute_text, *attribute_values = line.split(":")
                if attribute_values:
                    processed_values = set(map(lambda s: s.strip().replace(".", ""), attribute_values[0].split(",")))
                    attributes[attribute_text] = processed_values
            elif "," in line:
                labels = set(map(lambda s: s.strip(), line.split(",")))
            if "| columns" in line:
                parse_columns = True
    path_to_training_data = os.path.join("data", "car", "test.csv" if test else "train.csv")
    df = pd.read_csv(path_to_training_data, names=columns)
    return df, attributes, labels


def build_bank_features(test=False):
    path_to_columns = os.path.join("data", "bank", "columns.txt")
    columns = []
    attributes = dict()
    with open(path_to_columns, "r") as f:
        columns = list(map(lambda s: s.strip(), f.readlines()))
    data_filename = "test.csv" if test else "train.csv"
    path_to_data = os.path.join("data", "bank", data_filename)
    df = pd.read_csv(path_to_data, names=columns)
    for numeric_column_name in ("age", "balance", "day", "duration", "campaign", "pdays", "previous"):
        median = df[numeric_column_name].median()
        names = [f"<{median}", f"{median}+"]
        bins = [-np.inf, median, np.inf]
        df[numeric_column_name] = pd.cut(df[numeric_column_name], bins, labels=names)
    for categorical_column_name in ("job", "marital", "education", "contact", "month", "poutcome"):
        df[categorical_column_name] = df[categorical_column_name].astype("category")
    for bool_column_names in ("default", "housing", "loan", "label"):
        df[bool_column_names] = df[bool_column_names].replace(("yes", "no"), (True, False))
    attribute_keys = set(columns) - {"label"}
    attributes = dict(zip(attribute_keys, map(lambda c: list(df[c].unique()), attribute_keys)))
    return (df, attributes, set(df["label"].unique()))


if __name__ == "__main__":
    df, *_ = build_features("shy")
