import numpy as np
import matplotlib.pyplot as plt

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

x = np.linspace(-100, 100, 1000)
y = sigmoid(x)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim([-5, 5])
ax.grid(True)
ax.plot(x, y, color='blue')
plt.show()