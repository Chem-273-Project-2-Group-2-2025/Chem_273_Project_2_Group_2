import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Ecoli:
    def __init__(self, tsteps=2000, bacterium=3, dimensions=10, memory=4, squares=500, learn=50000, grad_std=6):
        self.tsteps = tsteps
        self.bacterium = bacterium
        self.memory = memory
        self.squares = squares
        self.plot_min = -dimensions / 2
        self.plot_max = dimensions / 2
        self.learn = learn
        self.grad_std = grad_std

        # Define grid
        self.grid_x = np.linspace(self.plot_min, self.plot_max, self.squares)
        self.grid_y = np.linspace(self.plot_min, self.plot_max, self.squares)

        # Create meshgrid for gradient
        self.grad_center_x = np.random.uniform(self.plot_min, self.plot_max)
        self.grad_center_y = np.random.uniform(self.plot_min, self.plot_max)
        
        self.x_grad, self.y_grad = np.meshgrid(self.grid_x, self.grid_y)
        self.gradient = norm.pdf(self.x_grad, self.grad_center_x, self.grad_std) * norm.pdf(self.y_grad, self.grad_center_y, self.grad_std)

        # Index-based positions (indices into grid_x, grid_y arrays)
        self.X_index = np.zeros((self.bacterium, self.tsteps), dtype=int)
        self.Y_index = np.zeros((self.bacterium, self.tsteps), dtype=int)

        # Initialize random starting positions within grid
        self.X_index[:, 0] = np.random.randint(0, self.squares, size=self.bacterium)
        self.Y_index[:, 0] = np.random.randint(0, self.squares, size=self.bacterium)

    def run_step(self, bac, step):
        # Calculate delta from memory steps ago
        delta_x = self.X_index[bac, step - 1] - self.X_index[bac, step - 1 - self.memory]
        delta_y = self.Y_index[bac, step - 1] - self.Y_index[bac, step - 1 - self.memory]
    
        x_curr = self.X_index[bac, step - 1]
        y_curr = self.Y_index[bac, step - 1]
    
        # Forward and backward positions clipped to grid bounds
        x_fwd = np.clip(x_curr + delta_x, 0, self.squares - 1)
        x_bwd = np.clip(x_curr - delta_x, 0, self.squares - 1)
        y_fwd = np.clip(y_curr + delta_y, 0, self.squares - 1)
        y_bwd = np.clip(y_curr - delta_y, 0, self.squares - 1)
    
        # Concentration values at those positions (note gradient[y, x])
        x_conc_fwd = self.gradient[y_curr, x_fwd]
        x_conc_bwd = self.gradient[y_curr, x_bwd]
        y_conc_fwd = self.gradient[y_fwd, x_curr]
        y_conc_bwd = self.gradient[y_bwd, x_curr]
    
        x_deriv = 0.0
        y_deriv = 0.0
    
        # Avoid division by zero
        if delta_x != 0:
            x_deriv = (x_conc_fwd - x_conc_bwd) / (2 * delta_x)
        if delta_y != 0:
            y_deriv = (y_conc_fwd - y_conc_bwd) / (2 * delta_y)
    
        # Calculate run jump using fixed learning rate
        x_run_exp = int(np.round(self.learn * x_deriv))
        y_run_exp = int(np.round(self.learn
                                 * y_deriv))
    
        # Update position clipped to grid boundaries
        new_x = np.clip(x_curr + x_run_exp, 0, self.squares - 1)
        new_y = np.clip(y_curr + y_run_exp, 0, self.squares - 1)

        self.X_index[bac, step] = new_x
        self.Y_index[bac, step] = new_y

    def tumble_step(self, bac, step):
        direction = np.random.choice([0, 1, 2, 3])
        x_prev = self.X_index[bac, step - 1]
        y_prev = self.Y_index[bac, step - 1]

        if direction == 0:  # up
            self.X_index[bac, step] = x_prev
            self.Y_index[bac, step] = (y_prev + 1) % self.squares
        elif direction == 1:  # down
            self.X_index[bac, step] = x_prev
            self.Y_index[bac, step] = (y_prev - 1) % self.squares
        elif direction == 2:  # left
            self.X_index[bac, step] = (x_prev - 1) % self.squares
            self.Y_index[bac, step] = y_prev
        elif direction == 3:  # right
            self.X_index[bac, step] = (x_prev + 1) % self.squares
            self.Y_index[bac, step] = y_prev

    def movement_steps(self):
        for step in range(1, self.tsteps):
            for bac in range(self.bacterium):
                # Every memory+1 steps run; otherwise tumble
                if (step + 1) % (self.memory + 1) == 0:
                    self.run_step(bac, step)
                else:
                    self.tumble_step(bac, step)

    def get_coordinates(self):
        X_coords = self.grid_x[self.X_index]
        Y_coords = self.grid_y[self.Y_index]
        return X_coords, Y_coords

    def plot_paths(self):
        X_coords, Y_coords = self.get_coordinates()
        plt.figure(figsize=(8, 8))

        # Plot gradient as background
        plt.imshow(self.gradient, extent=[self.plot_min, self.plot_max, self.plot_min, self.plot_max],
                   origin='lower', cmap='plasma', alpha=0.5)

        # Plot paths for each bacterium
        for b in range(self.bacterium):
            plt.plot(X_coords[b], Y_coords[b], marker='o', markersize=1, label=f'Bacterium {b+1}')

        plt.xlim(self.plot_min, self.plot_max)
        plt.ylim(self.plot_min, self.plot_max)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title("E. coli paths")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal')
        plt.show()



eco = Ecoli(tsteps=10000, bacterium=5, dimensions=20, squares=500)
eco.movement_steps()
eco.plot_paths()
