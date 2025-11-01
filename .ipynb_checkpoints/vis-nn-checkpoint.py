import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

df["input"] = df["input"].apply(ast.literal_eval)

X = np.array(df["input"].tolist(), dtype=int)
y = np.array(df["label"], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X0 = X_train[y_train == 0]
X1 = X_train[y_train == 1]

plt.figure(figsize=(6, 6))
plt.scatter(X0[:, 0], X0[:, 1], color='blue')
plt.scatter(X1[:, 0], X1[:, 1], color='orange')
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

np.random.seed(42)
W = np.random.randn(2, 1) * 0.01
b = 0.0

lr = 0.01
epochs = 10

losses = []

for epoch in range(epochs):

    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    
    loss = -np.mean(y * np.log(A + 1e-8) + (1 - y) * np.log(1 - A + 1e-8))
    losses.append(loss)

    dZ = A - y
    dW = np.dot(X.T, dZ) / len(X)
    db = np.mean(dZ)

    W -= lr * dW
    b -= lr * db
