#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 00:58:04 2025

@author: brenda
"""
import numpy as np
import matplotlib.pyplot as plt

def C_gaussian(x, y, x0=0.0, y0= 0.0, A = 1.0, sx= 20.0, sy=20.0):
    X,Y = x - x0, y - y0
    return A * np.exp(-0.5 * ((X/sx)**2 + (Y/sy)**2))


def C_sigmoid(x, y, x0= 0.0, y0= 0.0, A = 1.0, r0= 20.0, k= 5.0):
    r = np.sqrt((x-x0)**2 +(y-y0)**2)
    return A / (1.0 + np.exp(-(r - r0)/k))
    
class Sim_Ecoli: 
    def __init__(self, N= 100, total_time =500, bounds=(-50,50),\
        step_size= 1.0, seed= None, run_step= 2.0, dc_gain=0.4, \
            field_fun = C_sigmoid, field_params = None):
    
        self.N = int (N)
        self.total_time = int (total_time)
        self.bounds = tuple(bounds)
        self.step_size = float(step_size)
        self.lo, self.hi = self.bounds 
        self.run_step = float(run_step)
        self.dc_gain = float(dc_gain)
    
        #random number generator
        self.rng = np.random.default_rng(seed)
    
        #field choice
        self.field_fun = field_fun
        self.field_params = field_params
    
        #random initial positions
        xs = self.rng.uniform(self.lo, self.hi, size= self.N)
        ys= self.rng.uniform(self.lo, self.hi, size= self.N)
        self.positions = np.column_stack([xs,ys]).astype(float)
    
        #random initial directions
        angles= self.rng.uniform(0, 2 * np.pi, size = self.N)
        self.last_direction = np.column_stack([np.cos(angles), np.sin(angles)])
    
        self.phase = np.zeros(self.N, dtype = int)
        self.concentration_mark = np.zeros(self.N, dtype= float)
        self.trajectory = np.zeros((total_time +1, N, 2), dtype=float)
        self.trajectory[0] = self.positions
    
    def concentration(self, x, y):
        return self.field_fun(x,y, **self.field_params)
        
    def grad_point(self, x, y , eps= 1e-5):
        c0= self.concentration(x, y)
        gx= (self.concentration(x + eps, y)- c0)/ eps
        gy = (self.concentration(x, y + eps)- c0)/ eps
        return gx, gy
    
    def _clamp_i(self, i):
        self.positions[i,0]= np.clip(self.positions[i,0], self.lo, self.hi)
        self.positions[i, 1] = np.clip(self.positions[i,1], self.lo, self.hi)
        
    def random_walk(self):
        angle = self.rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype= float)
    
    def step(self):
        for i in range(self.N):
            if self.phase[i] == 0:
                self.concentration_mark[i] = self.concentration(self.positions[i,0], self.positions[i,1,])
            if self.phase[i] < 4:
                self.last_direction[i] = self.random_walk()
                self.positions[i] += self.step_size * self.last_direction[i]
                self._clamp_i(i)
                self.phase[i] += 1
            else:
                x,y = self.positions[i]
                gx, gy = self.grad_point(x, y)
                gnorm = np.hypot(gx, gy)
                if gnorm < 1e-12:
                    ux, uy = self.last_direction[i]
                else: 
                    ux, uy = gx/gnorm, gy/gnorm
                    self.last_direction[i]= np.array([ux, uy])
                    
                current_C = self.concentration(x, y)
                delta_C = current_C - self.concentration_mark[i]
                scale = 1.0 + self.dc_gain * np.tanh(delta_C)
                
                self.positions[i] += self.run_step * scale * np.array([ux, uy])
                self._clamp_i(i)
                self.phase[i]= 0
        return self.positions.copy()
    
    def run(self):
        for t in range(1, self.total_time + 1):
            self.step()
            self.trajectory[t] = self.positions
        return self.trajectory  

#plotting 

def plot_concentration(field_func, field_params, bounds= (-50,50), res= 200):
    lo, hi = bounds 
    xs= np.linspace(lo, hi, res)
    ys= np.linspace(lo, hi, res)
    XX, YY = np.meshgrid(xs, ys, indexing = 'xy')
    Z = field_func(XX, YY, **field_params)
    plt.figure(figsize=(7,6))
    plt.imshow(Z, extent = [lo, hi, lo, hi], origin= 'lower', aspect= 'equal')
    plt.colorbar(label= "Concentration")
    plt.title(f"{field_func.__name__} profile")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
def plot_histograms(trajectory, times, bounds= (-50,50)):
    for t in times:
        X_t= trajectory[t:, :, 0]
        plt.figure()
        plt.hist(X_t, bins= 40, range= bounds)
        plt.title(f"x histogram at t= {t}")
        plt.xlabel("x")
        plt.ylabel("count")
        plt.show()
        
def plot_trajectories(trajectory, K= 50, bounds=(50,50)):
    T, N, _ = trajectory.shape
    K= min(K, N)
    idx= np.linspace(0, -1, K, dtype=int) 
    plt.figure(figsize=(6,6))
    for i in idx:
        plt.plot(trajectory[:, i, 0], trajectory[:,i,1], lw=0.7, alpha= 0.7)
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.gca().set_aspect('equal', 'box')
    plt.title(f"Trajectories ({K}/ {N} shown")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
# main
if __name__ == "__main__":
    sigmoid_params = dict(x0=0.0, y0= 0.0, A=1.0, r0= 20.0, k = -5.0 )
    sim_sigmoid = Sim_Ecoli(N= 100, total_time= 500, bounds=(-50,50),\
                    seed=30, field_fun= C_sigmoid, field_params= sigmoid_params )
    traj_sigmoid = sim_sigmoid.run()
    
    plot_concentration(C_sigmoid, sigmoid_params, bounds=(-50, 50))
    plot_histograms(traj_sigmoid, times=[0,250, 500], bounds= (-50,50))
    plot_trajectories(traj_sigmoid, K=20, bounds=(-50, 50))