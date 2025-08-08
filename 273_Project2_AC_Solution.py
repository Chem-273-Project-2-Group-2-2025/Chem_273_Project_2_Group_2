#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:45:09 2025

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#simulation is working but there are some issues
#the derivatives from point to point are very small
#this means the learning rate needs to be very high, especially when ecoli are from from source
#this is causing some weird behavior where sometimes they find it sometimes not

## Check TO-DO comments throughout for what needs to get done!

class EColi:
    def __init__(self, N, num_ecoli=1, box_width=1, box_height=1, 
                 grid_res = 500, tumble_step=3, learning_rate=10, grad_seeds=1,
                 memory = 4, grad_type='gaussian'):
        
        #gradient descent parameters
        self.N = N
        self.tumble_step = tumble_step
        self.learning_rate = learning_rate
        self.memory = memory
        self.grad_type = grad_type
        
        #setup for grid
        self.x_bounds = box_width/2
        self.y_bounds = box_height/2
        self.grid_res = grid_res
        self.x_grid = np.linspace(-self.x_bounds, self.x_bounds, grid_res)
        self.y_grid = np.linspace(-self.y_bounds, self.y_bounds, grid_res)

# TO-DO: need to also have an option to seed all ecoli at the same starting position ################
        #setup for seeding ecoli at random positions
        self.num_ecoli = num_ecoli
        self.X = np.zeros((num_ecoli,N), dtype=int)
        self.Y = np.zeros((num_ecoli,N), dtype=int)
        self.X[:,0] = np.random.randint(0, len(self.x_grid), (num_ecoli,))
        self.Y[:,0] = np.random.randint(0, len(self.y_grid), (num_ecoli,))
        
        #setup for seeding gradient
        self.grad_seeds = grad_seeds
        self.X_grad, self.Y_grad = np.meshgrid(self.x_grid,self.y_grid)
        self.grad_xcenter = np.random.randint(0, len(self.x_grid), (self.grad_seeds,))
        self.grad_ycenter = np.random.randint(0, len(self.y_grid), (self.grad_seeds,))
        self.grad = np.zeros((len(self.x_grid),len(self.y_grid)))
        
        #used for conversion from index to grid for plot routine
        self.ecoli_grid_xpos = np.zeros((self.num_ecoli,self.N))
        self.ecoli_grid_ypos = np.zeros((self.num_ecoli,self.N))
        
        #Sum of squared gradients for adaptive learning rate (adagrad)
        self.Gt_x = np.zeros((self.num_ecoli,self.N))
        self.Gt_y = np.zeros((self.num_ecoli,self.N))
        
        
    def tumble_ecoli(self, n):
        
        #pick a number between -1 and 1, tumble_step number of times
        #sum across rows then execute move
        x_tumble = np.random.choice([-1,1], size = (self.num_ecoli, self.tumble_step)).sum(axis=1)
        y_tumble = np.random.choice([-1,1], size = (self.num_ecoli, self.tumble_step)).sum(axis=1)
        
        #if move is out of bounds then clip
        self.X[:,n+1] = np.clip((self.X[:,n] + x_tumble), a_min=0, a_max = len(self.x_grid)-1)
        self.Y[:,n+1] = np.clip((self.Y[:,n] + y_tumble), a_min=0, a_max = len(self.y_grid)-1)
        
        
    def run_ecoli(self,n):

        delta_x = self.X[:,n] - self.X[:,n-self.memory]
        delta_y = self.Y[:,n] - self.Y[:,n-self.memory]
        
        #if forward or backward delta is out of bounds then clip
        x_fwd = np.clip((self.X[:,n] + delta_x), a_min=0, a_max = len(self.x_grid)-1)
        x_bwd = np.clip((self.X[:,n] - delta_x), a_min=0, a_max = len(self.x_grid)-1)
        y_fwd = np.clip((self.Y[:,n] + delta_y), a_min=0, a_max = len(self.y_grid)-1)
        y_bwd = np.clip((self.Y[:,n] - delta_y), a_min=0, a_max = len(self.y_grid)-1)
        
        #calculate concentration from grad for fwd and bwd positions
        x_conc_fwd = self.grad[x_fwd, self.Y[:,n]]
        x_conc_bwd = self.grad[x_bwd, self.Y[:,n]]
        y_conc_fwd = self.grad[y_fwd, self.X[:,n]]
        y_conc_bwd = self.grad[y_bwd, self.X[:,n]]
        
        x_deriv = np.zeros((self.num_ecoli,))
        y_deriv = np.zeros((self.num_ecoli,))
        
        
        #bool mask to prevent divide by zero error if delta is zero
        x_mask = (delta_x != 0)
        x_deriv[x_mask] = (x_conc_fwd[x_mask] - x_conc_bwd[x_mask]) / (2 * delta_x[x_mask])
        
        y_mask = (delta_y != 0)
        y_deriv[y_mask] = (y_conc_fwd[y_mask] - y_conc_bwd[y_mask]) / (2 * delta_y[y_mask])
        
        #attempted to introduce adap[tive gradient (adagrad) approach for learning rate
        self.Gt_x[:,n] = x_deriv**2
        self.Gt_y[:,n] = y_deriv**2
        
        adapt_lr_x = self.learning_rate / np.sqrt((np.sum(self.Gt_x, axis=1)) + np.exp(-12))
        adapt_lr_y = self.learning_rate / np.sqrt((np.sum(self.Gt_x, axis=1)) + np.exp(-12))
        
        x_run_exp = np.round(adapt_lr_x * x_deriv)
        y_run_exp = np.round(adapt_lr_y * y_deriv)
        self.X[:,n+1] = np.clip((self.X[:,n] + x_run_exp), a_min=0, a_max = len(self.x_grid)-1)
        self.Y[:,n+1] = np.clip((self.Y[:,n] + y_run_exp), a_min=0, a_max = len(self.y_grid)-1)


        #old learning rate approach
        
        # x_run_exp = np.round(self.learning_rate * x_deriv) #can we lower learning rate as deriv increases to prevent overshoot near center?
        # y_run_exp = np.round(self.learning_rate * y_deriv)
        # self.X[:,n+1] = np.clip((self.X[:,n] + x_run_exp), a_min=0, a_max = len(self.x_grid)-1)
        # self.Y[:,n+1] = np.clip((self.Y[:,n] + y_run_exp), a_min=0, a_max = len(self.y_grid)-1)


# TO-DO: Try and figure out a vectorized way to do this ############################################
    def index_to_grid(self):
        
        for i in range(self.num_ecoli):
            for j in range(self.N):
                self.ecoli_grid_xpos[i,j] = self.x_grid[self.X[i,j]]
                self.ecoli_grid_ypos[i,j] = self.y_grid[self.Y[i,j]]
        


# TO-DO: Test different calculation approaches for conc. gradient other than pdf ###################

    def gaussian_grad(self, mesh, mu, sigma):
        return (1 / sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((mesh - mu)/(sigma))**2)
    
    def linear_grad(self, mesh, loc, slope, height):
        return -slope*np.abs(mesh - loc) + height
    
    def parabola_grad(self, mesh, loc, width, height):
        return -((mesh-loc)**2 / width**2) + height

#make a switch for grad_type
    def create_gradient(self, grad_type="gaussian"):
        #this loop allows you to seed multiple center points for the gradient
        for n in range(self.grad_seeds):
            match grad_type:
                case "gaussian":
                    self.grad += (self.gaussian_grad(self.X_grad, self.x_grid[self.grad_xcenter[n]], sigma=1)) * \
                                 (self.gaussian_grad(self.Y_grad, self.y_grid[self.grad_ycenter[n]], sigma=1))
                
                case "linear":
                    self.grad += (self.linear_grad(self.X_grad, self.x_grid[self.grad_xcenter[n]], slope=0.2, height=10)) +\
                                 (self.linear_grad(self.Y_grad, self.y_grid[self.grad_ycenter[n]], slope=0.2, height=10))
                                 
                case "parabola":
                    self.grad += (self.parabola_grad(self.X_grad, self.x_grid[self.grad_xcenter[n]], width=3, height=10)) + \
                                 (self.parabola_grad(self.Y_grad, self.y_grid[self.grad_ycenter[n]], width=3, height =10))
                         


# TO-DO: Add axis labels and improve look of plot #################################################
# Make it so first and lost point are marked with an X of different colors, add legend for marker #
    def plot_routine_track(self):
        
        plt.figure(figsize=(8,8))
        plt.contourf(self.X_grad, self.Y_grad, self.grad, levels=5, cmap='Grays')

        for n in range(self.num_ecoli):
            plt.plot(self.ecoli_grid_xpos[n,:],self.ecoli_grid_ypos[n,:], marker= 'o', markersize=1, linestyle='-', color='red')
        
        plt.xlim(-self.x_bounds,self.x_bounds)
        plt.ylim(-self.y_bounds,self.y_bounds)
        plt.show()


# TO-DO: Add plot_routine_histogram ###############################################################



# TO-DO: Create run simulation function ###########################################################
##       allow for arguments to get passed in for adjusting starting params #######################

N=1000

ecoli = EColi(N)
ecoli.create_gradient(ecoli.grad_type)

for n in range(N-1):
    if (n+1) % (ecoli.memory + 1) ==0:
         ecoli.run_ecoli(n)
    else:
         ecoli.tumble_ecoli(n)


ecoli.index_to_grid()
ecoli.plot_routine_track()