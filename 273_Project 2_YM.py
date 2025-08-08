import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 300
grid_size = 500
tumble = 5
learning = 2.0
memory = 4

x_grid = np.linspace(-10, 10, grid_size)
y_grid = np.linspace(-10, 10, grid_size)
Xg, Yg = np.meshgrid(x_grid, y_grid)

gradient = norm.pdf(Xg, 0, 2) * norm.pdf(Yg, 0, 2)

x_idx = int(0.85 * grid_size)
y_idx = int(0.6 * grid_size)
x_traj = [x_idx]
y_traj = [y_idx]


for step in range(N):
    if step < memory or (step % (memory + 1)) != 0:
        dx = np.random.choice([-1, 1], tumble).sum()
        dy = np.random.choice([-1, 1], tumble).sum()
    else:
        dx = x_traj[-1] - x_traj[-memory]
        dy = y_traj[-1] - y_traj[-memory]
        approxdCdx = (gradient[y_traj[-1], x_traj[-1] + dx] - gradient[y_traj[-1], x_traj[-1] - dx]) / (2 * dx)
        approxdCdy = (gradient[y_traj[-1] + dy, x_traj[-1]] - gradient[y_traj[-1] - dy, x_traj[-1]]) / (2 * dy)
        dx = int(np.sign(approxdCdx) * learning)
        dy = int(np.sign(approxdCdy) * learning)
        
    x_idx = np.clip(x_traj[-1] + dx, 0, grid_size - 1)
    y_idx = np.clip(y_traj[-1] + dy, 0, grid_size - 1)
    x_traj.append(x_idx)
    y_traj.append(y_idx)


x_vals = [x_grid[i] for i in x_traj]
y_vals = [y_grid[i] for i in y_traj]

plt.figure(figsize=(8,8))
plt.contourf(Xg, Yg, gradient, levels=20, cmap="viridis")
plt.plot(x_vals, y_vals, color='red')
plt.scatter(x_vals[0], y_vals[0], marker='x', color='black')
plt.title("E. coli random walk")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()