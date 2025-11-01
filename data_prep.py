import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

data_size = 10000
#blue poitns x and y
set1_x1 = np.random.randint(0, 50, data_size)
set1_x2 = np.random.randint(0, 100, data_size)

#orange points x and y
set2_x1 = np.random.randint(51, 100, data_size)
set2_x2 = np.random.randint(0, 100, data_size)

w1 = np.random.rand(data_size)
w2 = np.random.rand(data_size)

datas = []

for x1, x2 in zip(set1_x1, set1_x2):
    datas.append([[int(x1), int(x2)], 0])

for x1, x2 in zip(set2_x1, set2_x2):
    datas.append([[int(x1), int(x2)], 1])

df = pd.DataFrame(datas, columns=["input", "label"])
df.to_csv('dataset.csv', index=False)

points = np.array(df["input"].tolist())
labels = df["label"].to_numpy()

blue_points = points[labels == 0]
orange_points = points[labels == 1]

fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
ax1.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', label='Blue Points')
ax1.scatter(orange_points[:, 0], orange_points[:, 1], color='orange', label='Orange Points')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
plt.show()
