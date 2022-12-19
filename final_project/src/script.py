import pandas as pd

df = pd.read_csv("models/submission.csv", names=["ID", "Prediction"], header=0)
df["ID"] = df["ID"].apply(lambda x: x + 1)
df.set_index("ID")
df["Prediction"] = df["Prediction"].astype("int")
df.to_csv("models/prediction.csv", index=False)
