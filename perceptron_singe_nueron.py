import numpy as np

np.random.seed(42)

x1 = np.random.randint(-100, 100, 100)
labels = np.array([])

for x in x1:
    if x > 0:
        labels = np.append(labels, [1])
    else:
        labels = np.append(labels, [0])

x1.reshape(1, 100)
labels.reshape(1, 100)

w1 = np.random.randint(0, 100, 1) * 0.01
print("weight: ", w1)

epochs = 1
lr = 0.01

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def sigmoid_derivative(z):
    return z * (1 - z)

sig = np.array([])
epsilon = 1e-12

for epoch in range(epochs):
    for i in range(len(x1)):

        y = np.dot(w1, x1[i])
        z = sigmoid(y)
        z = np.where(z < epsilon, 0, 1)
        sig = np.append(sig, z)

        e = z - labels[i]
        

# for i in range(len(x1)):
#     print(f"x1 = {x1[i]}, label = {labels[i]}, weight = {w1}, sigmoid_value = {sig[i]}") 