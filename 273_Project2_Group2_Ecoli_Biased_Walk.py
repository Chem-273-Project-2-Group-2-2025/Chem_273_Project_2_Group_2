#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chem_273_Project 2_Group 2
"""

import numpy as np
import matplotlib.pyplot as plt

class EColi:
    def __init__(self, N, num_ecoli=1, box_length=20, grid_res = 500, tumble_step=0.3, 
                 learning_rate=1.5, memory = 4, grad_type='gaussian', origin_type='random'):
        
        #gradient descent parameters
        self.N = N
        self.tumble_step = tumble_step
        self.learning_rate = learning_rate
        self.memory = memory
        self.grad_type = grad_type
        
        #setup for grid
        self.box_length = box_length
        self.bounds = box_length/2
        self.grid_res = grid_res
        self.grid = np.linspace(-self.bounds, self.bounds, grid_res)

        #setup for E.coli seeds
        self.num_ecoli = num_ecoli
        self.X = np.zeros((num_ecoli,N))
        self.Y = np.zeros((num_ecoli,N))
        self.origin_type = origin_type
        if self.origin_type == 'random':
            self.X[:,0] = np.random.uniform(-self.bounds, 0, (num_ecoli,))
            self.Y[:,0] = np.random.uniform(-self.bounds, 0, (num_ecoli,))
        if self.origin_type == 'together':
            self.X[:,0] = -self.bounds*0.8
            self.Y[:,0] = -self.bounds*0.8

        
        #setup for seeding gradient
        self.X_grad, self.Y_grad = np.meshgrid(self.grid, self.grid)
        self.grad_xcenter = self.bounds*0.8
        self.grad_ycenter = self.bounds*0.8
        self.grad = np.zeros((len(self.grid),len(self.grid)))
        
        #Sum of squared gradients for adaptive learning rate (adagrad)
        self.Gt_x = np.zeros((self.num_ecoli,self.N))
        self.Gt_y = np.zeros((self.num_ecoli,self.N))
        self.adapt_lr_x = np.zeros((self.num_ecoli,self.N))
        self.adapt_lr_y = np.zeros((self.num_ecoli,self.N))
        
        
        #Distance index for histogram
        self.I = [1, 10, 50, 100, 1000]
        self.distance = np.zeros((self.num_ecoli,len(self.I)))
        
        #params for different gradient functions
        self.gaus_sigma = 10
        self.lin_slope = 0.5
        self.lin_height = 10
        self.parab_width = 1
        self.parab_height = 10
        

        
    def tumble_ecoli(self, n):
        
        dx = np.random.uniform(-self.tumble_step,self.tumble_step,(1, self.num_ecoli))
        dy = np.random.uniform(-self.tumble_step,self.tumble_step,(1, self.num_ecoli))
        
        self.X[:,n+1] = np.clip((self.X[:,n] + dx), a_min=-self.bounds, a_max=self.bounds)
        self.Y[:,n+1] = np.clip((self.Y[:,n] + dy), a_min=-self.bounds, a_max=self.bounds)
        
        
    def run_ecoli(self,n):

        delta_x = self.X[:,n] - self.X[:,n-self.memory]
        delta_y = self.Y[:,n] - self.Y[:,n-self.memory]
        
        # these are the finite differences used for calculating the derivatives
        # if forward or backward delta is out of bounds then clip
        x_fwd = np.clip((self.X[:,n] + delta_x), a_min=-self.bounds, a_max=self.bounds)
        x_bwd = np.clip((self.X[:,n] - delta_x), a_min=-self.bounds, a_max=self.bounds)
        y_fwd = np.clip((self.Y[:,n] + delta_y), a_min=-self.bounds, a_max=self.bounds)
        y_bwd = np.clip((self.Y[:,n] - delta_y), a_min=-self.bounds, a_max=self.bounds)
        
        # calculate concentration from grad for finite differences
        # specific conc_calc depends on gradient type specified by user
        x_conc_fwd = self.conc_calc(x_fwd, self.grad_xcenter)
        x_conc_bwd = self.conc_calc(x_bwd, self.grad_xcenter)
        y_conc_fwd = self.conc_calc(y_fwd, self.grad_ycenter)
        y_conc_bwd = self.conc_calc(y_bwd, self.grad_ycenter)
        
        x_deriv = np.zeros((self.num_ecoli,))
        y_deriv = np.zeros((self.num_ecoli,))
        
        # calculate partial derivatives by finite differences
        # bool mask to prevent divide by zero error if delta is zero
        x_mask = (delta_x != 0)
        x_deriv[x_mask] = (x_conc_fwd[x_mask] - x_conc_bwd[x_mask]) / (2 * delta_x[x_mask])
        
        y_mask = (delta_y != 0)
        y_deriv[y_mask] = (y_conc_fwd[y_mask] - y_conc_bwd[y_mask]) / (2 * delta_y[y_mask])
        
        # adaptive gradient (adagrad) approach for learning rate
        self.Gt_x[:,n] = x_deriv**2
        self.Gt_y[:,n] = y_deriv**2
        
        # this is the actual adagrad equation
        # the derivative for each previous run step is added to the sum
        # this causes the learning rate to reduce over time as the sum of derivatives grows
        self.adapt_lr_x[:,n] = self.learning_rate / np.sqrt((np.sum(self.Gt_x, axis=1)) + np.exp(-12))
        self.adapt_lr_y[:,n] = self.learning_rate / np.sqrt((np.sum(self.Gt_y, axis=1)) + np.exp(-12))
        
        x_run_exp = self.adapt_lr_x[:,n] * x_deriv
        y_run_exp = self.adapt_lr_y[:,n] * y_deriv
        
        # update the next iteration with new position after gradient "ascent"
        self.X[:,n+1] = np.clip((self.X[:,n] + x_run_exp), a_min=-self.bounds, a_max=self.bounds)
        self.Y[:,n+1] = np.clip((self.Y[:,n] + y_run_exp), a_min=-self.bounds, a_max=self.bounds)

    
    # calculates the distance from source for each E.coli for generating histograms
    def calculate_distance(self):

        dist_i = 0
        for i in self.I:
            self.distance[:,dist_i] = np.sqrt((self.grad_xcenter - self.X[:,i-1])**2 + \
                                    (self.grad_ycenter - self.Y[:,i-1])**2)
            dist_i += 1

    # define each gradient calculation
    # ecoli and gradient generation call on conc_calc which depending on gradient type
    # applies the correct function
    def gaussian_grad(self, value, mu):
        return (1 / self.gaus_sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((value - mu)/(self.gaus_sigma))**2)
    
    def linear_grad(self, value, loc):
        return -self.lin_slope*np.abs(value - loc) + self.lin_height
    
    def parabola_grad(self, value, loc):
        return -((value-loc)**2 / self.parab_width**2) + self.parab_height
    
    
    def conc_calc(self, val, grad_center):
        
        match self.grad_type:
            case "gaussian":
                result = self.gaussian_grad(val, grad_center)
            case "linear":
                result = self.linear_grad(val, grad_center)
            case "parabola":
                result = self.parabola_grad(val, grad_center)
                
        return result


    def create_gradient(self):
        
        #match case is like c++ switch, this is how the correct gradient is calculated
        match self.grad_type:
            case "gaussian":
                self.grad = (self.conc_calc(self.X_grad, self.grad_xcenter)) * \
                              (self.conc_calc(self.Y_grad, self.grad_ycenter))
                
            case "linear":
                self.grad = (self.conc_calc(self.X_grad, self.grad_xcenter)) +\
                              (self.conc_calc(self.Y_grad, self.grad_ycenter))
                                 
            case "parabola":
                self.grad = (self.conc_calc(self.X_grad, self.grad_xcenter)) + \
                              (self.conc_calc(self.Y_grad, self.grad_ycenter))
                         

    def plot_routine_track(self):
        
        fig, ax = plt.subplot_mosaic([['A'],['B']], layout="constrained", figsize=(8,8),
                                     gridspec_kw={'height_ratios': [7, 1]})
        
        ax['A'].contourf(self.X_grad, self.Y_grad, self.grad, levels=5, cmap='grey_r')

        for n in range(self.num_ecoli):
            ax['A'].plot(self.X[n,:],self.Y[n,:], marker= 'o', 
                     markerfacecolor='none', markeredgecolor='white', markersize=4, 
                     linestyle='-', color='white', alpha =0.5)
        
        for n in range(self.num_ecoli):
            ax['A'].scatter(self.X[n,0], self.Y[n,0], marker= 'x',
                        s=100, color='blue')
            ax['A'].scatter(self.X[n,-1], self.Y[n,-1], marker= 'x',
                        s=100, color='red')
        
        # mask for iterations where an adaptive learning rate was calculated
        # determine average lr for x and y and then normalize
        lr_x_mask = (self.adapt_lr_x != 0)
        lr_y_mask = (self.adapt_lr_y != 0)
        average_lr = (self.adapt_lr_x[lr_x_mask] + self.adapt_lr_y[lr_y_mask]) / 2
        num_lr = np.arange(0, len(average_lr), 1)
        min_lr = np.min(average_lr)
        max_lr = np.max(average_lr)
        norm_lr = (average_lr - min_lr) / (max_lr - min_lr)
        
        ax['A'].set_xlim([-self.bounds,self.bounds])
        ax['A'].set_ylim([-self.bounds,self.bounds])
        ax['A'].set_title(f"Biased Random Walk\nE.coli = {self.num_ecoli}, iterations = {self.N}, gradient type = {self.grad_type}, memory={self.memory}")
        ax['A'].set_xlabel("x position")
        ax['A'].set_ylabel("y position")
        
        # plots the adaptive learning rate over time
        # only plots up to 25 iterations because after that there is little change
        ax['B'].plot(num_lr, norm_lr, color='black')
        ax['B'].set_xlim([0, 25])
        ax['B'].set_ylim([0, None])
        ax['B'].tick_params(axis='x', bottom=False, labelbottom=False)
        ax['B'].set_xlabel("time")
        ax['B'].set_ylabel("Learning Rate")
        
        plt.show()
    
    
    def plot_routine_source_distance(self):
        
        # each plot will show the same kind of histogram at a different iteration
        # create a dict to hold the layout key and number of iterations for more
        # concise histogram plotting
        layout = ['A', 'B', 'C', 'D', 'E']
        fig_dict = dict(zip(self.I, layout))
        
        fig, ax = plt.subplot_mosaic([['A'],['B'],['C'],['D'],['E']], 
                                     layout="constrained", figsize=(8,12))
        
        dist_i = 0
        for i in self.I:
            ax[fig_dict[i]].hist(self.distance[:,dist_i], bins=200, range=(0, np.max(self.distance)), color='black')
            ax[fig_dict[i]].text(0.05,0.85,f"I = {i}", fontsize = 14, transform=ax[fig_dict[i]].transAxes)
            dist_i += 1
        
        ax['A'].set_title(f"Simulation of {self.num_ecoli} E.coli\nGradient = {self.grad_type}, Origin = {self.origin_type}", fontsize=20)
        ax['C'].set_ylabel("number of E.coli", fontsize=16)
        ax['E'].set_xlabel("distance from the source", fontsize=16)


# This is the primary simulation function
# generates an E.coli class with specificed parameters and performs specified plotting
# returns the class type for post simulation review if desired
def ecoli_simulation(num_ecoli, N=1000, mem = 4, origin='random', grad="gaussian", track_plot=False, dist_plot=False):
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
    
    # construct E.coli class for simulation and generate gradient
    ecoli = EColi(N, num_ecoli, memory=mem, grad_type=grad, origin_type=origin)
    ecoli.create_gradient()
    
    # perform N number of iterations and run depending on memory of E.coli
    for n in range(N-1):
        if (n+1) % (ecoli.memory + 1) ==0:
             ecoli.run_ecoli(n)
        else:
             ecoli.tumble_ecoli(n)
    
    # after simulation is complete calculate the distances from the source for
    # each iteration for histogram plot
    ecoli.calculate_distance()
    
    # perform plotting depending on what user has specified
    if track_plot:
        ecoli.plot_routine_track() 
    
    if dist_plot:
        ecoli.plot_routine_source_distance()

    return ecoli

# enter test group number to run test (1-4)
def run_test_cases(test):
    
    match test:
        
##### TEST 1 ##################################################################
    # testing our three different gradients, 1 E.coli each
    # 'together' origin starts E.coli at the same fixed point
    # generating 2d tracking plots
        case 1:
            t1_1 = ecoli_simulation(num_ecoli=1, grad='gaussian', origin='together', track_plot=True)
            t1_2 = ecoli_simulation(num_ecoli=1, grad='parabola', origin='together', track_plot=True)
            t1_3 = ecoli_simulation(num_ecoli=1, grad='linear', origin='together', track_plot=True)
    
##### TEST 2 ##################################################################
    # testing 10 E.coli generated on guassian gradient
    # comparing starting together vs random locations in lower left quadrant
    # generating 2d tracking plots
        case 2:
            t2_1 = ecoli_simulation(num_ecoli=10, grad='gaussian', origin='together', track_plot=True)
            t2_2 = ecoli_simulation(num_ecoli=10, grad='gaussian', origin='random', track_plot=True)

##### TEST 3 ##################################################################
    # testing location from source for 1,000 E.coli
    # comparting starting together vs random locations in lower left quadrant
    # comparing each gradient type
    # generating histograms at specific iterations
        case 3:
            t3_1 = ecoli_simulation(num_ecoli=1000, grad='gaussian', origin='together', dist_plot=True)
            t3_2 = ecoli_simulation(num_ecoli=1000, grad='gaussian', origin='random', dist_plot=True)
            t3_3 = ecoli_simulation(num_ecoli=1000, grad='parabola', origin='together', dist_plot=True)
            t3_4 = ecoli_simulation(num_ecoli=1000, grad='linear', origin='together', dist_plot=True)

##### TEST 4 ##################################################################
    # testing impact of memory on E.coli
        case 4:
            t4_1 = ecoli_simulation(num_ecoli=1, mem=3, grad='gaussian', origin='together', track_plot=True)
            t4_2 = ecoli_simulation(num_ecoli=1, mem=2, grad='gaussian', origin='together', track_plot=True)
            t4_3 = ecoli_simulation(num_ecoli=1, mem=1, grad='gaussian', origin='together', track_plot=True)
            
        case _:
            print("Not a valid test group")









