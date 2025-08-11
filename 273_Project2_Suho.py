import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Concentration Profiles -----------------------

class ConcentrationProfile:
    def gaussian(x, y, x0=0.0, y0=0.0, A=1.0, sx=6.0, sy=6.0):
        X, Y = x - x0, y - y0
        return A * np.exp(-0.5 * ((X/sx)**2 + (Y/sy)**2))

    def ring(x, y, x0=0.0, y0=0.0, A=1.0, r0=8.0, w=1.5):
        r = np.hypot(x - x0, y - y0)
        return A * np.exp(-0.5 * ((r - r0)/w)**2)

    def sigmoid(x, y, x0=0.0, y0=0.0, A=1.0, k=1.0, r_switch=8.0):
        r = np.hypot(x - x0, y - y0)
        return A / (1.0 + np.exp(k * (r - r_switch)))

# ----------------------- Simulation Class ----------------------------

class EColiSimulation:
    def __init__(
        self, 
        N_steps=200, 
        N_bacteria=100, 
        tumble_steps=4, 
        box_size=20, 
        tumble_size=0.5, 
        run_strength=1.0,
        concentration_func=None, 
        profile_kwargs=None, 
        seed=0
    ):
        self.rng = np.random.default_rng(seed)     # modern RNG
        self.N_steps = int(N_steps)
        self.N_bacteria = int(N_bacteria)
        self.tumble_steps = int(tumble_steps)      # memory length
        self.box_size = float(box_size)
        self.tumble_size = float(tumble_size)
        self.run_strength = float(run_strength)
        self.concentration_func = concentration_func or ConcentrationProfile.gaussian
        self.profile_kwargs = profile_kwargs or {}

        # pos: shape (N_steps, N_bacteria, 2)  (t, i, [x,y])
        self.pos = np.zeros((self.N_steps, self.N_bacteria, 2), dtype=float)
        # random start anywhere in the box
        self.pos[0, :, 0] = self.rng.uniform(-self.box_size/2, self.box_size/2, self.N_bacteria)
        self.pos[0, :, 1] = self.rng.uniform(-self.box_size/2, self.box_size/2, self.N_bacteria)

        # concentration trace (optional but handy)
        self.conc = np.zeros((self.N_steps, self.N_bacteria), dtype=float)
        self.conc[0] = self._C(self.pos[0,:,0], self.pos[0,:,1])

    # concentration wrapper
    def _C(self, x, y):
        return self.concentration_func(x, y, **self.profile_kwargs)

    def _clip_box(self, xy):
        np.clip(xy, -self.box_size/2, self.box_size/2, out=xy)

    def run(self):
        mem = self.tumble_steps
        for t in range(1, self.N_steps):
            # --- TUMBLE: 3 out of 4 steps ---
            if t < mem or (t % mem) != 0:
                angle = self.rng.uniform(0, 2*np.pi, self.N_bacteria)
                step = np.stack([np.cos(angle), np.sin(angle)], axis=1) * self.tumble_size
                self.pos[t] = self.pos[t-1] + step
            else:
                # --- RUN: use memory between t-1 and t-mem to estimate gradient sign ---
                t_now = t-1
                t_past = max(0, t - mem)

                disp = self.pos[t_now] - self.pos[t_past]       # (N,2)
                norm = np.linalg.norm(disp, axis=1) + 1e-12     # avoid /0
                dirn = disp / norm[:, None]                     # unit displacement

                c_now  = self._C(self.pos[t_now,:,0],  self.pos[t_now,:,1])
                c_past = self._C(self.pos[t_past,:,0], self.pos[t_past,:,1])

                # finite difference along recent path; sign decides up/down gradient
                dc = (c_now - c_past) / norm                    # (N,)
                sign = np.sign(dc)[:, None]                     # (N,1) in {-1,0,1}

                run_vec = dirn * sign * self.run_strength
                self.pos[t] = self.pos[t_now] + run_vec

            # keep inside box
            self._clip_box(self.pos[t])
            self.conc[t] = self._C(self.pos[t,:,0], self.pos[t,:,1])

    # ----------------------- Plotting ----------------------------------

    def plot_trajectory(self, idx=0, show_gradient=True):
        plt.figure(figsize=(7,7))
        if show_gradient:
            xs = np.linspace(-self.box_size/2, self.box_size/2, 200)
            ys = np.linspace(-self.box_size/2, self.box_size/2, 200)
            X, Y = np.meshgrid(xs, ys)
            Z = self._C(X, Y)
            plt.contourf(X, Y, Z, levels=12, cmap='Greys')

        xy = self.pos[:, idx, :]
        plt.plot(xy[:,0], xy[:,1], 'w.-', lw=2, ms=4)
        plt.xlim(-self.box_size/2, self.box_size/2)
        plt.ylim(-self.box_size/2, self.box_size/2)
        plt.gca().set_aspect("equal", "box")
        plt.title(f'Trajectory (idx={idx})')
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()

    def plot_all_histograms(self, t_indices):
        fig, axs = plt.subplots(len(t_indices), 1, figsize=(8, 2.2*len(t_indices)), sharex=True)
        # assume source at (0,0) for these profiles; adjust if needed
        r = np.hypot(self.pos[:,:,0], self.pos[:,:,1])  # (T, N)
        for i, t_idx in enumerate(t_indices):
            axs[i].hist(r[t_idx], bins=60, range=(0, self.box_size/2), color='k')
            axs[i].set_ylabel('count')
            axs[i].set_title(f"N={self.N_bacteria}, t={t_idx}")
        axs[-1].set_xlabel('distance from source')
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # try different N as in the spec
    for N_bacteria in [10, 100, 1000]:
        sim = EColiSimulation(
            N_steps=1200,
            N_bacteria=N_bacteria,
            tumble_steps=4,
            box_size=30,
            tumble_size=0.35,
            run_strength=0.9,
            concentration_func=ConcentrationProfile.gaussian,
            profile_kwargs={"x0":0.0, "y0":0.0, "A":1.0, "sx":6.0, "sy":6.0},
            seed=7
        )
        sim.run()
        if N_bacteria == 10:
            sim.plot_trajectory(0)
        sim.plot_all_histograms([1,10,50,100,1000])