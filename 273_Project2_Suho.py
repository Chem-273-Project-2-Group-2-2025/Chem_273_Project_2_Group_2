# Chem 273 Project 2, E. coli chemotaxis
# Authors: your team names here

import numpy as np
import matplotlib.pyplot as plt

class ConcentrationProfile:
    # Gaussian peak centered at x0, y0
    @staticmethod
    def gaussian(x, y, x0=0.0, y0=0.0, A=1.0, sx=6.0, sy=6.0):
        X = x - x0
        Y = y - y0
        return A * np.exp(-0.5 * ((X / sx) ** 2 + (Y / sy) ** 2))

    # Ring shaped peak around radius r0
    @staticmethod
    def ring(x, y, x0=0.0, y0=0.0, A=1.0, r0=8.0, w=1.5):
        r = np.hypot(x - x0, y - y0)
        return A * np.exp(-0.5 * ((r - r0) / w) ** 2)

    # Sigmoid front that increases toward the center
    @staticmethod
    def sigmoid(x, y, x0=0.0, y0=0.0, A=1.0, k=1.0, r_switch=8.0):
        r = np.hypot(x - x0, y - y0)
        return A / (1.0 + np.exp(k * (r - r_switch)))

class EColiSimulation:
    """
    Biased random walk with four step memory.
    Tumble for three steps.
    On the fourth step, compare concentration at t and t minus 4.
    Move along the recent displacement direction if the change was positive.
    Move against it if the change was negative.
    """

    def __init__(
        self,
        N_steps=1200,
        N_bacteria=100,
        box_size=30.0,
        tumble_size=0.35,
        run_strength=0.9,
        memory=4,
        concentration_func=None,
        profile_kwargs=None,
        seed=7
    ):
        # Configuration
        self.N_steps = int(N_steps)
        self.N_bacteria = int(N_bacteria)
        self.box = float(box_size)
        self.tumble_size = float(tumble_size)
        self.run_strength = float(run_strength)
        self.memory = int(memory)

        # Concentration field
        self.C = concentration_func or ConcentrationProfile.gaussian
        self.kw = profile_kwargs or {}

        # RNG
        self.rng = np.random.default_rng(seed)

        # Positions, shape (T, N, 2)
        self.pos = np.zeros((self.N_steps, self.N_bacteria, 2), dtype=float)
        self.pos[0, :, 0] = self.rng.uniform(-self.box / 2, self.box / 2, self.N_bacteria)
        self.pos[0, :, 1] = self.rng.uniform(-self.box / 2, self.box / 2, self.N_bacteria)

        # Concentration history, optional for analysis
        self.conc = np.zeros((self.N_steps, self.N_bacteria), dtype=float)
        self.conc[0] = self._C(self.pos[0, :, 0], self.pos[0, :, 1])

    # Concentration wrapper
    def _C(self, x, y):
        return self.C(x, y, **self.kw)

    # Keep agents in the box
    def _clip_box(self, xy):
        np.clip(xy, -self.box / 2, self.box / 2, out=xy)

    # One simulation step
    def _step(self, t):
        if t < self.memory or (t % self.memory) != 0:
            # Tumble
            ang = self.rng.uniform(0, 2 * np.pi, self.N_bacteria)
            step = np.stack([np.cos(ang), np.sin(ang)], axis=1) * self.tumble_size
            self.pos[t] = self.pos[t - 1] + step
        else:
            # Run, use memory t minus 4
            t_now = t - 1
            t_past = max(0, t - self.memory)

            disp = self.pos[t_now] - self.pos[t_past]
            norm = np.linalg.norm(disp, axis=1) + 1e-12
            dirn = disp / norm[:, None]

            c_now = self._C(self.pos[t_now, :, 0], self.pos[t_now, :, 1])
            c_past = self._C(self.pos[t_past, :, 0], self.pos[t_past, :, 1])

            dc = (c_now - c_past) / norm
            sign = np.sign(dc)[:, None]

            run_vec = dirn * sign * self.run_strength
            self.pos[t] = self.pos[t_now] + run_vec

        self._clip_box(self.pos[t])
        self.conc[t] = self._C(self.pos[t, :, 0], self.pos[t, :, 1])

    # Full simulation
    def run(self):
        for t in range(1, self.N_steps):
            self._step(t)
        return self.pos

    # Plot the concentration field as background
    def _plot_concentration(self, ax, levels=12, title_suffix=""):
        xs = np.linspace(-self.box / 2, self.box / 2, 200)
        ys = np.linspace(-self.box / 2, self.box / 2, 200)
        X, Y = np.meshgrid(xs, ys)
        Z = self._C(X, Y)
        cntr = ax.contourf(X, Y, Z, levels=levels, cmap="Greys")
        ax.set_aspect("equal", "box")
        ax.set_xlim(-self.box / 2, self.box / 2)
        ax.set_ylim(-self.box / 2, self.box / 2)
        ax.set_title(f"Concentration and paths{title_suffix}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return cntr

    # Multiple trajectory snapshots in one call
    def plot_trajectories(self, times, max_traces=100, levels=12):
        cols = len(times)
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6), squeeze=False)
        subset = min(max_traces, self.N_bacteria)
        show_idx = self.rng.choice(self.N_bacteria, size=subset, replace=False)

        for j, t in enumerate(times):
            ax = axes[0, j]
            self._plot_concentration(ax, levels=levels, title_suffix=f", t={t}")
            for k in show_idx:
                ax.plot(self.pos[:t + 1, k, 0], self.pos[:t + 1, k, 1],
                        "-", lw=0.9, c="white", alpha=0.9, marker="o", ms=1.6, mec="k", mfc="w")

        plt.tight_layout()
        plt.show()

    # Stacked histograms of distance from source at chosen times
    def plot_distance_histograms(self, times):
        r = np.hypot(self.pos[:, :, 0], self.pos[:, :, 1])
        fig, axes = plt.subplots(len(times), 1, figsize=(8, 2.2 * len(times)), sharex=True)

        for i, t in enumerate(times):
            ax = axes[i]
            ax.hist(r[t], bins=60, range=(0, self.box / 2), color="k")
            ax.set_ylabel("count")
            ax.set_title(f"Distance from source, N={self.N_bacteria}, t={t}")

        axes[-1].set_xlabel("distance from source")
        plt.tight_layout()
        plt.show()

# Simple battery of tests to satisfy the checklist
def run_project2_tests():
    # Three profiles
    profiles = [
        ("gaussian", {"x0": 0.0, "y0": 0.0, "A": 1.0, "sx": 6.0, "sy": 6.0}),
        ("ring", {"x0": 0.0, "y0": 0.0, "A": 1.0, "r0": 8.0, "w": 1.5}),
        ("sigmoid", {"x0": 0.0, "y0": 0.0, "A": 1.0, "k": 1.0, "r_switch": 8.0}),
    ]

    # Three N values
    Ns = [10, 100, 1000]

    # Times for plots
    traj_times = [200, 600, 1100]
    hist_times = [1, 10, 50, 100, 1000]

    for prof_name, prof_kw in profiles:
        for N in Ns:
            sim = EColiSimulation(
                N_steps=1200,
                N_bacteria=N,
                box_size=30.0,
                tumble_size=0.35,
                run_strength=0.9,
                memory=4,
                concentration_func=getattr(ConcentrationProfile, prof_name),
                profile_kwargs=prof_kw,
                seed=7
            )
            sim.run()
            sim.plot_trajectories(times=traj_times, max_traces=100)
            if N >= 50:
                sim.plot_distance_histograms(times=hist_times)

if __name__ == "__main__":
    run_project2_tests()