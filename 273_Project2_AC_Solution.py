#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:45:09 2025

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

class EColi:
    def __init__(self, N, num_ecoli=1, box_length=1, grid_res = 350, tumble_step=3, 
                 learning_rate=25, memory = 4, grad_type='gaussian', origin_type='random'):
        
        #gradient descent parameters
        self.N = N
        self.tumble_step = tumble_step
        self.learning_rate = learning_rate
        self.memory = memory
        self.grad_type = grad_type
        
        #setup for grid
        self.box_length = box_length
        self.x_bounds = box_length/2
        self.y_bounds = box_length/2
        self.grid_res = grid_res
        self.x_grid = np.linspace(-self.x_bounds, self.x_bounds, grid_res)
        self.y_grid = np.linspace(-self.y_bounds, self.y_bounds, grid_res)

# TO-DO: need to also have an option to seed all ecoli at the same starting position ################
        #setup for seeding ecoli at random positions
        self.num_ecoli = num_ecoli
        self.X = np.zeros((num_ecoli,N), dtype=int)
        self.Y = np.zeros((num_ecoli,N), dtype=int)
        self.origin_type = origin_type
        if self.origin_type == 'random':
            self.X[:,0] = np.random.randint(0.1*len(self.x_grid), 0.3*len(self.x_grid), (num_ecoli,))
            self.Y[:,0] = np.random.randint(0.1*len(self.y_grid), 0.3*len(self.y_grid), (num_ecoli,))
        if self.origin_type == 'together':
            self.X[:,0] = int(0.1*len(self.x_grid))
            self.Y[:,0] = int(0.1*len(self.y_grid))
            # self.X[:,0] = np.array((num_ecoli,0.25*len(self.x_grid)), dtype=int)
            # self.Y[:,0] = np.array((num_ecoli,0.25*len(self.x_grid)), dtype=int)
        
        #setup for seeding gradient
        self.X_grad, self.Y_grad = np.meshgrid(self.x_grid,self.y_grid)
        self.grad_xcenter = int(grid_res*0.8)
        self.grad_ycenter = int(grid_res*0.8)
        self.grad_xcenter_grid = self.x_grid[self.grad_xcenter]
        self.grad_ycenter_grid = self.y_grid[self.grad_ycenter]
        self.grad = np.zeros((len(self.x_grid),len(self.y_grid)))
        
        #used for conversion from index to grid for plot routine
        self.ecoli_grid_xpos = np.zeros((self.num_ecoli,self.N))
        self.ecoli_grid_ypos = np.zeros((self.num_ecoli,self.N))
        
        #Sum of squared gradients for adaptive learning rate (adagrad)
        self.Gt_x = np.zeros((self.num_ecoli,self.N))
        self.Gt_y = np.zeros((self.num_ecoli,self.N))
        
        #Distance index for histogram
        self.I = [1, 10, 50, 100, 1000]
        self.distance = np.zeros((self.num_ecoli,len(self.I)))
        
        
    def tumble_ecoli(self, n):
        
        #pick a number between -1 and 1, tumble_step number of times
        #sum across rows then execute move
        x_tumble = np.random.choice([-2,2], size = (self.num_ecoli, self.tumble_step)).sum(axis=1)
        y_tumble = np.random.choice([-2,2], size = (self.num_ecoli, self.tumble_step)).sum(axis=1)
        
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
        
        #attempted to introduce adaptive gradient (adagrad) approach for learning rate
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
    
    def calculate_distance(self):

        dist_i = 0
        for i in self.I:
            self.distance[:,dist_i] = np.sqrt((self.grad_xcenter_grid - self.ecoli_grid_xpos[:,i-1])**2 + \
                                    (self.grad_ycenter_grid - self.ecoli_grid_xpos[:,i-1])**2)
            dist_i += 1


    def gaussian_grad(self, mesh, mu, sigma):
        return (1 / sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((mesh - mu)/(sigma))**2)
    
    def linear_grad(self, mesh, loc, slope, height):
        return -slope*np.abs(mesh - loc) + height
    
    def parabola_grad(self, mesh, loc, width, height):
        return -((mesh-loc)**2 / width**2) + height


    def create_gradient(self):

        match self.grad_type:
            case "gaussian":
                self.grad = (self.gaussian_grad(self.X_grad, self.x_grid[self.grad_xcenter], sigma=1)) * \
                              (self.gaussian_grad(self.Y_grad, self.y_grid[self.grad_ycenter], sigma=1))
                
            case "linear":
                self.grad = (self.linear_grad(self.X_grad, self.x_grid[self.grad_xcenter], slope=0.5, height=10)) +\
                              (self.linear_grad(self.Y_grad, self.y_grid[self.grad_ycenter], slope=0.5, height=10))
                                 
            case "parabola":
                self.grad = (self.parabola_grad(self.X_grad, self.x_grid[self.grad_xcenter], width=1.5, height=10)) + \
                              (self.parabola_grad(self.Y_grad, self.y_grid[self.grad_ycenter], width=1.5, height =10))
                         

# TO-DO: Add axis labels and improve look of plot #################################################
# Make it so first and lost point are marked with an X of different colors, add legend for marker #
    def plot_routine_track(self):
        
        plt.figure(figsize=(8,8))
        plt.contourf(self.X_grad, self.Y_grad, self.grad, levels=5, cmap='grey_r')

        for n in range(self.num_ecoli):
            plt.plot(self.ecoli_grid_xpos[n,:],self.ecoli_grid_ypos[n,:], marker= 'o', 
                     markerfacecolor='none', markeredgecolor='white', markersize=4, 
                     linestyle='-', color='white', alpha =0.5)
        
        for n in range(self.num_ecoli):
            plt.scatter(self.ecoli_grid_xpos[n,0], self.ecoli_grid_ypos[n,0], marker= 'x',
                        s=100, color='blue')
            plt.scatter(self.ecoli_grid_xpos[n,-1], self.ecoli_grid_ypos[n,-1], marker= 'x',
                        s=100, color='red')
        
        plt.xlim(-self.x_bounds,self.x_bounds)
        plt.ylim(-self.y_bounds,self.y_bounds)
        plt.show()
    
    def plot_routine_source_distance(self):
        
        layout = ['A', 'B', 'C', 'D', 'E']
        fig_dict = dict(zip(self.I, layout))
        
        fig, ax = plt.subplot_mosaic([['A'],['B'],['C'],['D'],['E']], 
                                     layout="constrained", figsize=(8,12))
        
        dist_i = 0
        for i in self.I:
            ax[fig_dict[i]].hist(self.distance[:,dist_i], bins=200, range=(0, self.box_length), color='black')
            ax[fig_dict[i]].text(0.05,0.85,f"I = {i}", fontsize = 14, transform=ax[fig_dict[i]].transAxes)
            dist_i += 1
        
        ax['A'].set_title(f"Simulation of {self.num_ecoli} E.coli", fontsize=20)
        ax['C'].set_ylabel("number of E.coli", fontsize=16)
        ax['E'].set_xlabel("distance from the source", fontsize=16)



def ecoli_simulation(num_ecoli, N=1000, origin='random', grad="gaussian", track_plot=False, dist_plot=False):
    """
    Parameters
    ----------
    num_ecoli : int, required
        Number of E.coli to seed.
    N : int, optional
        Set number of iterations. The default is 1000.
    origin : str, optional
        E.coli seeding ('random', 'together'). The default is 'random'.
    grad : str, optional
        Gradient type ('guassian', 'linear', 'parabola'). The default is "gaussian".
    track_plot : bool, optional
        Run tracking plot routine. The default is False.
    dist_plot : bool, optional
        Run distance from source plot routine. The default is False.
    """

    ecoli = EColi(N, num_ecoli, grad_type=grad, origin_type=origin)
    ecoli.create_gradient()
    
    for n in range(N-1):
        if (n+1) % (ecoli.memory + 1) ==0:
             ecoli.run_ecoli(n)
        else:
             ecoli.tumble_ecoli(n)
    
    ecoli.index_to_grid()
    ecoli.calculate_distance()

    if track_plot:
        ecoli.plot_routine_track()
    
    if dist_plot:
        ecoli.plot_routine_source_distance()

    return ecoli

e = ecoli_simulation(num_ecoli=1, origin='random', track_plot=True)