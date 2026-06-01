import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../data/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_X,
    y,
    test_size=0.2,
    random_state=0
)

model = KNeighborsClassifier(
    n_neighbors=3
)

model.fit(X_train, y_train)