import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import ast

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

df = pd.read_csv("dataset.csv")
df["input"] = df["input"].apply(ast.literal_eval)

x = np.array(df["input"].tolist(), dtype=int)
y = np.array(df["label"], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

np.random.seed(42)
W = np.random.randn(1, 2) * 0.01
b = 0.0
lr = 0.01
epochs = 100

transformed = []

for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i].reshape(2, 1)
        y_true = y_train[i]

        z = np.dot(W, x) + b
        a = sigmoid(z)

        epsilon = 1e-8
        a = np.clip(a, epsilon, 1 - epsilon)

        loss = -(y_true * np.log(a) + (1 - y_true) * np.log(1 - a))

        dz = a - y_true
        dW = np.dot(dz, x.T)
        db = dz

        W -= lr * dW
        b -= lr * db 

    if epoch == epochs - 1:
        for x in X_train:
            x = x.reshape(2, 1)
            z = np.dot(W, x) + b
            transformed.append(z.flatten())


with open("transformed_data.pkl", "wb") as f:
    pickle.dump({
        "originals": X_test,
        "labels": y_test,
        "W": W,
        "b": b, 
        "X_train": X_train,
        "y_train": y_train
    }, f)


