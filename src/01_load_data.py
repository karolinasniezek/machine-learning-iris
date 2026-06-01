import pandas as pd

df = pd.read_csv("../data/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]