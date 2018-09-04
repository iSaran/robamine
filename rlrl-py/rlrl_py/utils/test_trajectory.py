import trajectory
import numpy as np

T = trajectory.Trajectory([10, 15], [-10, 20])
T2 = trajectory.Trajectory([15, 20], [20, -10])
t = np.linspace(10, 20, 200)
y = []
for k in t:
    if k < 15:
        y.append(T.pos(k))
    else:
        y.append(T2.pos(k))

import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, y)
plt.show()

