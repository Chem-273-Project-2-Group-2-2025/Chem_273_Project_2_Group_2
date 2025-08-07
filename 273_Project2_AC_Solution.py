#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:45:09 2025

@author: alex
"""

#simulate biased random walk of e coli
#gradient descent along a concentration gradient
#randomly seed N number of e. coli

#4 random walk tumble steps
#calcculate concentration gradient from t vs t-4deltat
#move along gradient, run
#repeat

#plot histogram of E.coli in xy plane for different t
#plost motion in xy plane for different t
    #visualize concentration gradient, meshgrid
    #allow for different concentration profiles
#write as an actual software package

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class EColi:
    def __init__(self, N, num_ecoli=1, box_width=20, box_height=20, 
                 tumble_step=0.5, learning_rate=500, grad_seeds=1):
        
        #gradient descent parameters
        self.N = N
        self.tumble_step = tumble_step
        self.learning_rate = learning_rate
        
        #setup for seeding ecoli at random positions
        self.num_ecoli = num_ecoli
        self.grad_seeds = grad_seeds
        self.x_bounds = box_width/2
        self.y_bounds = box_height/2
        
        self.X = np.zeros((num_ecoli,N))
        self.Y = np.zeros((num_ecoli,N))
        self.X[:,0] = np.random.uniform(-self.x_bounds, self.x_bounds, (num_ecoli,))
        self.Y[:,0] = np.random.uniform(-self.y_bounds, self.y_bounds, (num_ecoli,))
        
        #setup for seeding gradient
        x = np.arange(-self.x_bounds, self.x_bounds, 1)
        y = np.arange(-self.y_bounds, self.y_bounds, 1)
        self.X_grad, self.Y_grad = np.meshgrid(x,y)
        self.grad_xcenter = np.random.uniform(-self.x_bounds, self.x_bounds, (self.grad_seeds,))
        self.grad_ycenter = np.random.uniform(-self.y_bounds, self.y_bounds, (self.grad_seeds,))
        self.pdf = np.zeros((box_width,box_height))

    
    def tumble_ecoli(self, n):
        dx = np.random.uniform(-self.tumble_step,self.tumble_step,(1, self.num_ecoli))
        dy = np.random.uniform(-self.tumble_step,self.tumble_step,(1, self.num_ecoli))
        
        self.X[:,n+1] = self.X[:,n] + dx
        self.Y[:,n+1] = self.Y[:,n] + dy
        
        
    def run_ecoli(self,n):
        #create class variable for memory, default=4
        memory = 4
        
        delta_x = self.X[:,n] - self.X[:,n-memory]
        delta_y = self.Y[:,n] - self.Y[:,n-memory]
        
        #pdf array for gradient is not continous
        #need to find which bin each x and y for my calculation fall into
        #rounded down to get a whole number index value
        #so if x is 2.1 it will find pdf value in index 2, etc.
        
        ctr_x = np.clip(np.floor(self.X[:,n]), a_min=-9, a_max=9)
        fwd_x = np.clip(np.floor(self.X[:,n] + delta_x), a_min=-9, a_max=9)
        bwd_x = np.clip(np.floor(self.X[:,n] - delta_x), a_min=-9, a_max=9)
        ctr_y = np.clip(np.floor(self.Y[:,n]), a_min=-9, a_max=9)
        fwd_y = np.clip(np.floor(self.Y[:,n] + delta_y), a_min=-9, a_max=9)
        bwd_y = np.clip(np.floor(self.Y[:,n] - delta_y), a_min=-9, a_max=9)
        
        #range has same index values as gradient array
        #eventually replace this with the range we used to make the meshgrid#################################################TO-DO
        x_range = np.arange(-self.x_bounds, self.x_bounds, 1)
        y_range = np.arange(-self.y_bounds, self.y_bounds, 1)
        
        
        x_index = np.zeros((self.num_ecoli,3), dtype=int)
        y_index = np.zeros((self.num_ecoli,3), dtype=int)
        
        x_conc = np.zeros((self.num_ecoli,2))
        y_conc = np.zeros((self.num_ecoli,2))
        
        deriv_x = np.zeros((self.num_ecoli,1))
        deriv_y = np.zeros((self.num_ecoli,1))
        
        for i in range(self.num_ecoli):
            x_index[i,0] = (np.argwhere(x_range == ctr_x[i])).item()
            y_index[i,0] = (np.argwhere(y_range == ctr_y[i])).item()
            x_index[i,1] = (np.argwhere(x_range == fwd_x[i])).item()
            y_index[i,1] = (np.argwhere(y_range == fwd_y[i])).item()
            x_index[i,2] = (np.argwhere(x_range == bwd_x[i])).item()
            y_index[i,2] = (np.argwhere(y_range == bwd_y[i])).item()
        
            x_conc[i,0] = self.pdf[x_index[i,1], y_index[i,0]]
            x_conc[i,1] = self.pdf[x_index[i,2], y_index[i,0]]
            
            y_conc[i,0] = self.pdf[y_index[i,1], x_index[i,0]]
            y_conc[i,1] = self.pdf[y_index[i,2], x_index[i,0]]
            
            deriv_x[i] = (x_conc[i,0] - x_conc[i,1]) / 2*delta_x[i]
            deriv_y[i] = (y_conc[i,0] - y_conc[i,1]) / 2*delta_y[i]
            
        #flipped sign from neg to ascend gradient
        self.X[:,n+1] = self.X[:,n] + self.learning_rate*deriv_x[:,0]
        self.Y[:,n+1] = self.Y[:,n] + self.learning_rate*deriv_y[:,0]

    def create_gradient(self):

        for n in range(self.grad_seeds):
            self.pdf += ((norm.pdf(self.X_grad, loc = self.grad_xcenter[n], scale=5) * \
                   norm.pdf(self.Y_grad, loc=self.grad_ycenter[n], scale=5)))
        
        #self.pdf *= 1000

        
    def plot_routine_track(self):
        
        plt.figure(figsize=(8,8))
        plt.contourf(self.X_grad, self.Y_grad, self.pdf, levels=10)
        
        for n in range(self.num_ecoli):
            plt.plot(self.X[n,:],self.Y[n,:], marker= 'o', markersize=1, linestyle='-', color='red')
        
        plt.xlim(-9,9)
        plt.ylim(-9,9)
        plt.show()
        
        
        
def grad_descent_simulation(N = 100):
    
    ecoli = EColi(N)
    ecoli.create_gradient()
    
    for n in range(N-1):
        if (n+1) % 5 ==0:
             ecoli.run_ecoli(n) #swapped these
        else:
             ecoli.tumble_ecoli(n)
    
    #ecoli.run_ecoli(4)
    ecoli.plot_routine_track()
        
grad_descent_simulation()
        
        