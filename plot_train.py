import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle

with open('transformed_data.pkl', 'rb') as f:
    data = pickle.load(f)

x = data['X_train']
labels = data['y_train']

blue = []
orange = []

for i in range(len(labels)):
    if labels[i] == 0:
        blue.append(x[i])
    else:
        orange.append(x[i])

blue = np.array(blue)
orange = np.array(orange)

s0_x = blue[:, 0]
s0_y = blue[:, 1]

s1_x = orange[:, 0]
s1_y = orange[:, 1]

fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

ax1.scatter(s0_x, s0_y, color='blue')
ax1.scatter(s1_x, s1_y, color='orange')
plt.show()
