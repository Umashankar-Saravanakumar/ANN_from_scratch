import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle

with open('transformed_data.pkl', 'rb') as f:
    data = pickle.load(f)

x = data['originals']
labels = data['labels']
W = data['W']
b = data['b']

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

# ax1.scatter(s0_x, s0_y, color='blue')
# ax1.scatter(s1_x, s1_y, color='orange')

# trans_w1 = []
# for iter in range(len(x)):
#     i_ = x[iter][0] * round(W[0][0], 2)
#     i_ = round(i_, 2)
#     i_ = np.array(i_)
#     trans_w1.append([i_, 0])
# trans_w1 = np.array(trans_w1)

# trans_w2 = []
# for iter in range(len(x)):
#     j_ = x[:, 1][iter] * round(W[0][1], 2)
#     j_ = round(j_, 2)
#     j_ = np.array(j_)
#     trans_w2.append([j_, 0])
# trans_w2 = np.array(trans_w2)

y = []
for i in range(len(x)):
    y.append(np.dot(W, x[i]) + b[0])

y = np.array(y)

transformed = []
for i in range(len(y)):
    transformed.append([y[:, 0][i], 0])

transformed = np.array(transformed)

sigmoid = []
epsilon = 1e-10
for i in range(len(y)):
    val = y[i]
    if isinstance(val, np.ndarray) or isinstance(val, list):
        val = val[0]
    z = 1 / (1 + np.exp(-val))
    z = int(z >= 0.5)
    sigmoid.append([z, 0])

sigmoid = np.array(sigmoid)
#print(sigmoid)

x1, x2, y1, y2 = -100, 110, -5, 110

def update(frame):

    global x1, x2, y1, y2

    #print(frame)

    if frame<99:
        ax1.clear()
        # ax2.clear()
        # ax3.clear()

        t = frame / 99

        current_points_g1 = x + t * (transformed - x)
        # current_points_g2 = x + t * (trans_w2 - x)
        # current_points_g3 = x + t * (transformed - x)

        ax1.set_xlim(-200, 200)
        ax1.set_ylim(-5, 110)

        # ax2.set_xlim(-100, 100)
        # ax2.set_ylim(-100, 100)

        # ax3.set_xlim(-100, 100)
        # ax3.set_ylim(-100, 100)

        blue_points_g1 = current_points_g1[labels == 0]
        orange_points_g1 = current_points_g1[labels == 1]

        # blue_points_g2 = current_points_g2[labels == 0]
        # orange_points_g2 = current_points_g2[labels == 1]

        # blue_points_g3 = current_points_g3[labels == 0]
        # orange_points_g3 = current_points_g3[labels == 1]

        ax1.scatter(blue_points_g1[:, 0], blue_points_g1[:, 1], color='blue')
        ax1.scatter(orange_points_g1[:, 0], orange_points_g1[:, 1], color='orange')

        ax1.grid(True)

        # ax2.scatter(blue_points_g2[:, 0], blue_points_g2[:, 1], color='blue')
        # ax2.scatter(orange_points_g2[:, 0], orange_points_g2[:, 1], color='orange')

        # ax3.scatter(blue_points_g3[:, 0], blue_points_g3[:, 1], color='blue')
        # ax3.scatter(orange_points_g3[:, 0], orange_points_g3[:, 1], color='orange')

    else:
        ax1.clear()
        t = (frame - 100) / 99
        current_points = transformed + t * (sigmoid - transformed)
        #print(current_points)
        # if x1 < -0.25:
        #     x1+=1
        #     if x1 > -.25:
        #         x1 = -.25

        # if x2 > 1:
        #     x2-=1.5
        #     if x2 < 1:
        #         x2 = 1
        print(current_points)

        ax1.set_xlim(-200, 200)
        ax1.set_ylim(-5, 110)
        blue_points = current_points[labels == 0]
        orange_points = current_points[labels == 1]
        ax1.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')
        ax1.scatter(orange_points[:, 0], orange_points[:, 1], color='orange')
        ax1.grid(True)


ani = animation.FuncAnimation(fig, update, frames=200, interval=1, repeat=False)
plt.show()