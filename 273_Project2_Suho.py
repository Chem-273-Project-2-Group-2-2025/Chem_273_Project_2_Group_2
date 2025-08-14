import numpy as np
import matplotlib.pyplot as plt

class ConcentrationProfile:
    @staticmethod
    def gaussian(x, y, x0=0.0, y0=0.0, A=1.0, sx=6.0, sy=6.0):
        X, Y = x - x0, y - y0
        return A * np.exp(-0.5 * ((X/sx)**2 + (Y/sy)**2))
    @staticmethod
    def ring(x, y, x0=0.0, y0=0.0, A=1.0, r0=8.0, w=1.5):
        r = np.hypot(x - x0, y - y0)
        return A * np.exp(-0.5 * ((r - r0)/w)**2)
    @staticmethod
    def sigmoid(x, y, x0=0.0, y0=0.0, A=1.0, k=1.0, r_switch=8.0):
        r = np.hypot(x - x0, y - y0)
        return A / (1.0 + np.exp(k * (r - r_switch)))

class EColiSimulation:
    def __init__(self,
                 N_steps=200, N_bacteria=100, tumble_steps=4,
                 box_size=20, tumble_size=0.5, run_strength=1.0,
                 concentration_func=None, profile_kwargs=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.N_steps = int(N_steps)
        self.N_bacteria = int(N_bacteria)
        self.memory = int(tumble_steps)
        self.box = float(box_size)
        self.tumble_size = float(tumble_size)
        self.run_strength = float(run_strength)
        self.C = concentration_func or ConcentrationProfile.gaussian
        self.kw = profile_kwargs or {}
        self.pos = np.zeros((self.N_steps, self.N_bacteria, 2))
        self.pos[0,:,0] = self.rng.uniform(-self.box/2, self.box/2, self.N_bacteria)
        self.pos[0,:,1] = self.rng.uniform(-self.box/2, self.box/2, self.N_bacteria)
        self.conc = np.zeros((self.N_steps, self.N_bacteria))
        self.conc[0] = self.C(self.pos[0,:,0], self.pos[0,:,1], **self.kw)

    def _clip(self, xy):
        np.clip(xy, -self.box/2, self.box/2, out=xy)

    def run(self):
        for t in range(1, self.N_steps):
            if t < self.memory or (t % self.memory) != 0:
                ang = self.rng.uniform(0, 2*np.pi, self.N_bacteria)
                step = np.stack([np.cos(ang), np.sin(ang)], axis=1) * self.tumble_size
                self.pos[t] = self.pos[t-1] + step
            else:
                t_now = t-1
                t_past = max(0, t - self.memory)
                disp = self.pos[t_now] - self.pos[t_past]
                norm = np.linalg.norm(disp, axis=1) + 1e-12
                dirn = disp / norm[:,None]
                c_now = self.C(self.pos[t_now,:,0], self.pos[t_now,:,1], **self.kw)
                c_past = self.C(self.pos[t_past,:,0], self.pos[t_past,:,1], **self.kw)
                dc = (c_now - c_past) / norm
                run_vec = dirn * np.sign(dc)[:,None] * self.run_strength
                self.pos[t] = self.pos[t_now] + run_vec
            self._clip(self.pos[t])
            self.conc[t] = self.C(self.pos[t,:,0], self.pos[t,:,1], **self.kw)

    def plot_trajectory(self, idx=0, show_gradient=True):
        plt.figure(figsize=(7,7))
        if show_gradient:
            xs = np.linspace(-self.box/2, self.box/2, 200)
            ys = np.linspace(-self.box/2, self.box/2, 200)
            X, Y = np.meshgrid(xs, ys)
            Z = self.C(X, Y, **self.kw)
            plt.contourf(X, Y, Z, levels=12, cmap='Greys')
        xy = self.pos[:, idx, :]
        plt.plot(xy[:,0], xy[:,1], 'w.-', lw=2, ms=4)
        plt.xlim(-self.box/2, self.box/2)
        plt.ylim(-self.box/2, self.box/2)
        plt.gca().set_aspect('equal', 'box')
        plt.title('Trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def plot_all_histograms(self, t_indices):
        r = np.hypot(self.pos[:,:,0], self.pos[:,:,1])
        fig, axs = plt.subplots(len(t_indices), 1, figsize=(8, 2.2*len(t_indices)), sharex=True)
        for i, t_idx in enumerate(t_indices):
            axs[i].hist(r[t_idx], bins=60, range=(0, self.box/2), color='k')
            axs[i].set_ylabel('count')
            axs[i].set_title(f'N={self.N_bacteria}, t={t_idx}')
        axs[-1].set_xlabel('distance from source')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    sim = EColiSimulation(
        N_steps=1200, N_bacteria=800, tumble_steps=4,
        box_size=30, tumble_size=0.35, run_strength=0.9,
        concentration_func=ConcentrationProfile.gaussian,
        profile_kwargs={'x0':0,'y0':0,'A':1,'sx':6,'sy':6},
        seed=7
    )
    sim.run()
    sim.plot_trajectory(0)
    sim.plot_all_histograms([1,10,50,100,1000])