"""build_features.py contains methods for extracting the features from datasets"""
import os
import pandas as pd


def build_features(dataset):
    path_to_data_desc = os.path.join("data", dataset, "data-desc.txt")
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
                    processed_values = set(map(lambda s: s.strip().replace(".", ""), attribute_values))
                    attributes[attribute_text] = processed_values
            elif "," in line:
                labels = set(map(lambda s: s.strip(), line.split(",")))
            if "| columns" in line:
                parse_columns = True
    path_to_training_data = os.path.join("data", dataset, "train.csv")
    df = pd.read_csv(path_to_training_data, names=columns)
    return df, attributes, labels


if __name__ == "__main__":
    print(build_features("car"))
