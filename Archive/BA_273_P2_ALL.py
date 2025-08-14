#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 00:58:04 2025

@author: brenda
"""
import numpy as np
import matplotlib.pyplot as plt


def C_gaussian(x, y, x0=0.0, y0=0.0, A=1.0, sx=10.0, sy=10.0):
    X, Y = x - x0, y - y0
    return A * np.exp(-0.5 * ((X/sx)**2 + (Y/sy)**2))


def G_gaussian(x, y, x0=0.0, y0=0.0, A=1.0, sx=10.0, sy=10.0):
    C = C_gaussian(x, y, x0, y0, A, sx, sy)
    gx = -(x-x0) * C / (sx**2)
    gy = -(y-y0) * C / (sy**2)
    return gx, gy


def C_linear(x, y, ax=1.0, ay=0.0, b=0.0):
    return ax * x + ay * y + b


def G_linear(x, y, ax=1.0, ay=0.0, b=0.0):
    return ax, ay


def C_sigmoid_radial(x, y, x0=0.0, y0=0.0, A=1.0, r0=10.0, k=5.0):
    r = np.hypot(x-x0, y-y0)
    return A / (1.0 + np.exp(-(r - r0)/k))


def G_sigmoid_radial(x, y, x0=0.0, y0=0.0, A=1.0, r0=10.0, k=5.0):
    X, Y = x-x0, y-y0
    r = np.hypot(X, Y)
    sigma = 1.0 / (1.0 + np.exp(-(r-r0)/k))
    dCdr = (A / k) * sigma * (1.0 - sigma)
    eps = 1e-12
    ux = np.where(r > eps, X / r, 0.0)
    uy = np.where(r > eps, Y / r, 0.0)
    return dCdr * ux, dCdr * uy


PROFILES= {
    "gaussian": {
        "C": lambda x, y, p: C_gaussian(x, y, **p),
        "G": lambda x, y, p: G_gaussian(x, y, **p),
        "params": dict(x0=0.0, y0=0.0, A=1.0, sx=12.0, sy=12.0),
    },
    "linear": {
        "C": lambda x, y, p: C_linear(x, y, **p),
        "G": lambda x, y, p: G_linear(x, y, **p),
        "params": dict(ax=1.0, ay=0.0, b=0.0),
    },
    "sigmoid_radial": {
        "C": lambda x, y, p: C_sigmoid_radial(x, y, **p),
        "G": lambda x, y, p: G_sigmoid_radial(x, y, **p),
        "params": dict(x0=0.0, y0=0.0, A=1.0, r0=10.0, k=0.5),
    },
}


class Sim_Ecoli:
    def __init__(self, N=100, total_time=1000, bounds=(-50, 50),
                 step_size=1.0, seed=42, run_step=1.2, dc_gain=0.2,
                 profile="gaussian", profile_params=None):

        self.N = int(N)
        self.total_time = int(total_time)
        self.bounds = tuple(bounds)
        self.step_size = float(step_size)
        self.lo, self.hi = self.bounds
        self.run_step = float(run_step)
        self.dc_gain = float(dc_gain)

        # random number generator
        self.rng = np.random.default_rng(seed)

        # field choice
        if profile not in PROFILES:
            raise ValueError(f"unknown profile '{profile}'.\
                             Choose from {list(PROFILES)}")
        self._C = PROFILES[profile]["C"]
        self._G = PROFILES[profile]["G"]
        self.p = dict(PROFILES[profile]["params"])
        if profile_params:
            self.p.update(profile_params)

        # random initial positions
        xs = self.rng.uniform(self.lo, self.hi, size=self.N)
        ys = self.rng.uniform(self.lo, self.hi, size=self.N)
        self.positions = np.column_stack([xs, ys]).astype(float)

        # random initial directions
        angles = self.rng.uniform(0, 2 * np.pi, size=self.N)
        self.last_direction = np.column_stack([np.cos(angles), np.sin(angles)])

        self.phase = np.zeros(self.N, dtype=int)
        self.concentration_mark = np.zeros(self.N, dtype=float)
        
        self.trajectory = np.zeros((total_time + 1, N, 2), dtype=float)
        self.trajectory[0] = self.positions
        

    def _clamp_i(self, i):
        self.positions[i, 0] = np.clip(self.positions[i, 0], self.lo, self.hi)
        self.positions[i, 1] = np.clip(self.positions[i, 1], self.lo, self.hi)

    def random_walk(self):
        angle = self.rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=float)

    def step(self):
        for i in range(self.N):
            if self.phase[i] == 0:
                self.concentration_mark[i] = self._C\
                    (self.positions[i, 0], self.positions[i, 1], self.p)
                
            if self.phase[i] < 4:
                self.last_direction[i] = self.random_walk()
                self.positions[i] += self.step_size * self.last_direction[i]
                self._clamp_i(i)
                self.phase[i] += 1
            else:
                x, y = self.positions[i]
                gx, gy = self._G(x, y, self.p)
                gnorm = np.hypot(gx, gy)
                if gnorm < 1e-12:
                    ux, uy = self.last_direction[i]
                else:
                    ux, uy = gx/gnorm, gy/gnorm
                    self.last_direction[i] = np.array([ux, uy])

                delta_C = self._C(x, y, self.p) - self.concentration_mark[i]
                scale = 1.0 + self.dc_gain * np.tanh(delta_C)

                self.positions[i] += self.run_step * scale * np.array([ux, uy])
                self._clamp_i(i)
                self.phase[i] = 0
        return self.positions.copy()

    def run(self):
        for t in range(1, self.total_time + 1):
            self.step()
            self.trajectory[t] = self.positions
        return self.trajectory

# plotting functions
def plot_concentration(profile, bounds=(-50, 50), res=200, profile_params=None, ax= None):
    if ax is None: ax=plt.gca()
    lo, hi = bounds
    xs = np.linspace(lo, hi, res)
    ys = np.linspace(lo, hi, res)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    spec = PROFILES[profile]
    p = dict(spec["params"])
    if profile_params: p.update(profile_params)
    Z = spec["C"](XX, YY, p)
    im= ax.imshow(Z, extent=[lo, hi, lo, hi], origin='lower', aspect='equal')
    ax.set_title(profile)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im

def plot_trajectories(trajectory, K=50, bounds=(50, 50), ax = None, title= None):
    if ax is None: ax = plt.gca()
    T, N, _ = trajectory.shape
    K = min(K, N)
    idx = np.linspace(0, N-1, K, dtype=int)
    for i in idx:
        ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], lw=0.7, alpha=0.7)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect('equal', adjustable= 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title: ax.set_title(title)

def plot_histograms_at_t(trajectory, t, bounds=(-50, 50), ax= None, title= None):
        if ax is None: ax= plt.gca()
        X_t = trajectory[t, :, 0]
        ax.hist(X_t, bins=40, range=bounds)
        ax.set_xlabel("x")
        ax.set_ylabel("count")
        if title: ax.set_title(title)

# main
if __name__ == "__main__":
    N, TOTAL, bounds, seed = 100, 1000, (-50, 50), 42
    
    step_size = 1
    run_step = 1.5
    dc_gain =  0.2
    
    setups = [
        ("gaussian",    {}),
        ("linear",      {"ax": 1.0, "ay": 0.0, "b": 0.0}),
        ("sigmoid_radial", {"x0": 0.0, "y0": 0.0,
         "A": 1.0, "r0": 20.0, "k": 5.0}),
    ]

    #clamped and periodic
    sims = []
    
    for name, params in setups:
        sim = Sim_Ecoli(N=N, total_time=TOTAL, bounds=bounds, seed=seed,\
                        step_size= step_size, run_step= run_step, dc_gain= dc_gain,\
                            profile=name, profile_params=params)
        sims.append((name, params, sim.run()))
        
        

    # plot concentration maps
    fig, axes = plt.subplots(1, 3, figsize=(15,5), constrained_layout= True)
    for ax, (name, params, _) in zip(axes, sims):
        im= plot_concentration(name, bounds= bounds, profile_params= params, ax=ax)
        fig.colorbar(im, ax=ax, shrink=0.85)
    plt.show()

    # trajectectories
    fig, axes= plt.subplots(1,3, figsize=(15,5), constrained_layout= True)
    for ax, (name, _, trajectory) in zip (axes, sims):
        plot_trajectories(trajectory, K=30, bounds=bounds, ax=ax, title=f"{name} trajectories")
    plt.show()

    # Final-time histograms
    fig, axes= plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    for ax, (name, _, trajectory) in zip(axes, sims):
        plot_histograms_at_t(trajectory, t=TOTAL, bounds=bounds, ax=ax, title=f"{name} x-hist @ T")
    plt.show()
