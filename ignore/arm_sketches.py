import numpy as np
import matplotlib.pyplot as plt

def turn_matrix_2d(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

r_i = [
    np.array([1.92,0.0]),
    np.array([1.95, 0.0]),
    np.array([1.1, 0.0])
]

thetas = [2*np.pi * 38.5/360, 2*np.pi * 34/360, 2*np.pi * 49/360]

configs = list(zip(r_i, thetas))
for j in range(2, -1, -1):
    sub_configs = configs[j:]
    s_i = []
    for i in range(len(sub_configs) + 1):
        s = np.array([0.0,0.0])
        for (r,theta) in sub_configs[:i][::-1]:
            s = turn_matrix_2d(theta) @ (s + r)
        s_i.append(s)

    print(s_i)


    plt.scatter([x[0] for x in s_i], [x[1] for x in s_i])
    plt.axis('equal')
    plt.show()
