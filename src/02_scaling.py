import pandas as pd

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)