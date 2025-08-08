import numpy as np
import matplotlib.pyplot as plt

class EColiSimulation:
    def __init__(
            self, N_steps=200, N_bacteria=100, tumble_steps=4, 
            box_size=20, tumble_size=0.5, run_strength=1.0,
            concentration_func=None, profile_kwargs=None, seed=0):
        np.random.seed(seed)
        self.N_steps = N_steps
        self.N_bacteria = N_bacteria
        self.tumble_steps = tumble_steps
        self.box_size = box_size
        self.tumble_size = tumble_size
        self.run_strength = run_strength
        self.concentration_func = concentration_func or ConcentrationProfile.gaussian
        self.profile_kwargs = profile_kwargs or {}
        self.init_positions()
        
    def init_positions(self):
        # Start all bacteria at one side of the box (e.g., lower left)
        self.x = np.zeros((self.N_bacteria, self.N_steps))
        self.y = np.zeros((self.N_bacteria, self.N_steps))
        # Can randomize starting points if you want
        self.x[:, 0] = np.random.uniform(-self.box_size/2, self.box_size/2, self.N_bacteria)
        self.y[:, 0] = np.random.uniform(-self.box_size/2, self.box_size/2, self.N_bacteria)
        
    def grad_concentration(self, x, y):
        # Compute local gradient numerically (central difference)
        eps = 1e-2
        cx = self.concentration_func(x, y, **self.profile_kwargs)
        dc_dx = (self.concentration_func(x+eps, y, **self.profile_kwargs) - cx) / eps
        dc_dy = (self.concentration_func(x, y+eps, **self.profile_kwargs) - cx) / eps
        return np.array([dc_dx, dc_dy])
    
    def run(self):
        memory = self.tumble_steps  # Number of steps between run decisions
        for t in range(1, self.N_steps):
            if t < memory:
                # Initial steps are just random walk (tumble)
                angle = np.random.uniform(0, 2*np.pi, self.N_bacteria)
                dx = self.tumble_size * np.cos(angle)
                dy = self.tumble_size * np.sin(angle)
            elif t % memory != 0:
                # Continue tumbling
                angle = np.random.uniform(0, 2*np.pi, self.N_bacteria)
                dx = self.tumble_size * np.cos(angle)
                dy = self.tumble_size * np.sin(angle)
            else:
                # "Run": move up gradient of concentration
                dx = np.zeros(self.N_bacteria)
                dy = np.zeros(self.N_bacteria)
                for i in range(self.N_bacteria):
                    x_mem = self.x[i, t-memory]
                    y_mem = self.y[i, t-memory]
                    x_now = self.x[i, t-1]
                    y_now = self.y[i, t-1]
                    # Compute direction of concentration change over last memory steps
                    grad = self.grad_concentration(x_now, y_now)
                    norm = np.linalg.norm(grad)
                    if norm > 0:
                        grad = grad / norm  # normalize
                    dx[i] = self.run_strength * grad[0]
                    dy[i] = self.run_strength * grad[1]
            # Update positions
            self.x[:, t] = self.x[:, t-1] + dx
            self.y[:, t] = self.y[:, t-1] + dy

    def plot_trajectory(self, idx=0, show_gradient=True):
        # Plot a single trajectory with the gradient background
        plt.figure(figsize=(8,8))
        xg, yg = np.meshgrid(
            np.linspace(-self.box_size/2, self.box_size/2, 100),
            np.linspace(-self.box_size/2, self.box_size/2, 100)
        )
        if show_gradient:
            z = self.concentration_func(xg, yg, **self.profile_kwargs)
            plt.contourf(xg, yg, z, levels=10, cmap='gray')
        plt.plot(self.x[idx, :], self.y[idx, :], 'w.-', lw=2, markersize=6)
        plt.title('Single E. coli Trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def plot_histogram(self, t_idx=None):
        # Plot histogram of distances from source at given t_idx
        t_idx = t_idx if t_idx is not None else -1
        center = np.array([0, 0])  # adjust if you move source
        dists = np.sqrt((self.x[:,t_idx]-center[0])**2 + (self.y[:,t_idx]-center[1])**2)
        plt.figure(figsize=(8,4))
        plt.hist(dists, bins=40, color='k')
        plt.title(f"Histogram of E. coli distances at t={t_idx}")
        plt.xlabel('distance from source')
        plt.ylabel('number of E. coli')
        plt.show()

    def plot_all_histograms(self, t_indices):
        # Plot histogram panel as in your sample image
        fig, axs = plt.subplots(len(t_indices), 1, figsize=(8, 2.5*len(t_indices)), sharex=True)
        center = np.array([0,0])
        for i, t_idx in enumerate(t_indices):
            dists = np.sqrt((self.x[:,t_idx]-center[0])**2 + (self.y[:,t_idx]-center[1])**2)
            axs[i].hist(dists, bins=40, color='k')
            axs[i].set_ylabel('number of E. coli')
            axs[i].set_title(f"N={self.N_bacteria}, t={t_idx}")
        axs[-1].set_xlabel('distance from source')
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Experiment with N = 1, 10, 50, 100, 1000
    for N_bacteria in [1, 10, 50, 100, 1000]:
        sim = EColiSimulation(N_steps=200, N_bacteria=N_bacteria, run_strength=1.0)
        sim.run()
        if N_bacteria == 1:
            sim.plot_trajectory(0)
        # Try various time indices for histograms
        sim.plot_all_histograms([50, 100, 150, 199])