import numpy as np 
import matplotlib.pyplot as plt 

# gradient descent along concentration gradient, N =10, 100, 1000..
# procedure - four subsequent time steps: random walk (tumble) 
# calculate concentration gradient from t vs t- 4 delta t
# move along gradient(run)
# repeat
# plot histogram of locations of E.coli for different t
# plot motion of E.coli in x-y plane for different t 
# visualize concentration profile 
# experiment with three concentration profiles

def C_gaussian(x, y, x0= 0.0, y0= 0.0, A= 1.0, sx= 20.0, sy= 20.0):
    X, Y = x - x0, y - y0
    return A * np.exp(-0.5 * ((X/sx)**2 +(Y/sy)**2))

class Sim_EColi:
    def __init__(self, N= 100, total_time= 1000, bounds= (-50, 50),\
                  step_size = 1.0, seed=None, x0= 0.0, y0= 0.0, A= 1.0, sx= 20, sy= 20.0,\
                run_step= 2.0, dc_gain=0.4):
        
        self.N = int(N)
        self.total_time = int(total_time)
        self.bounds = tuple(bounds) 
        self.step_size = float(step_size)
        self.lo, self.hi = self.bounds 
        self.run_step = float(run_step)
        self.dc_gain = float(dc_gain)

        #random number generation
        self.rng = np.random.default_rng(seed)

        #gaussian parameters
        self.x0, self.y0, self.A, self.sx, self.sy = float(x0), float(y0), float(A), float(sx), float(sy)

        #initialize positions
        xs= self.rng.uniform(self.lo, self.hi, size = self.N)
        ys= self.rng.uniform(self.lo, self.hi, size = self.N)
        self.positions = np.column_stack([xs, ys]).astype(float)
        
        #last direction (unit vectors)
        angles = self.rng.uniform(0, 2 * np.pi, size= self.N)      #generate random angles
        self.last_direction = np.column_stack([np.cos(angles), np.sin(angles)])   #convert angles to unit vectors

        #phase: 0 to 3 = tumble; 4 = decision/run 
        self.phase = np.zeros(self.N, dtype= int)    

        #initial concentration 
        self.concentration_mark = np.zeros(self.N, dtype= float)              

        #trajectory storage: (T+1, N, 2)
        self.trajectory = np.zeros((total_time + 1, N, 2), dtype= float)
        self.trajectory[0] = self.positions

    #concentration field
    def concentration_gaussian(self, x, y):
        return C_gaussian(x, y, self.x0, self.y0, self.A, self.sx, self.sy)
    
    def grad_point(self, x, y):
        C = self.concentration_gaussian(x,y)
        gx = -(x - self.x0) * C / (self.sx**2)
        gy = -(y - self.y0) * C / (self.sy**2)
        return gx, gy
    
    def _clamp_i(self, i):                                       #ensure ecoli stay in bounds
        self.positions[i, 0] = np.clip(self.positions[i, 0], self.lo, self.hi)
        self.positions[i, 1] = np.clip(self.positions[i,1], self.lo, self.hi)
    
    def random_walk(self):                          #random unit vectors
        angle = self.rng.uniform(0, 2 * np.pi)      
        return np.array([np.cos(angle), np.sin(angle)], dtype = float)

    def step(self):
        for i in range(self.N):
            if self.phase[i] == 0:       #phase 0, concentration at start
                self.concentration_mark[i] = self.concentration_gaussian(self.positions[i,0],\
                                                                         self.positions[i,1])
                
            if self.phase[i] < 4:                          #tumble, pick new random direction, move one unit
                self.last_direction[i] = self.random_walk()
                self.positions[i] += self.step_size * self.last_direction[i]
                self._clamp_i(i)
                self.phase[i] += 1

            else: 
                x,y = self.positions[i]   #run along spatial gradient
                gx, gy = self.grad_point(x,y)
                gnorm = np.hypot(gx,gy)

                if gnorm < 1e-12:
                    ux, uy = self.last_direction[i]
                
                else:
                    ux, uy = gx/gnorm, gy/gnorm 
                    self.last_direction[i]= np.array([ux,uy])

                current_C = self.concentration_gaussian(x,y) #compare current concentration vs marked concentration,run
                delta_C = current_C - self.concentration_mark[i]
                scale = 1.0 + self.dc_gain * np.tanh(delta_C)

                self.positions[i] += self.run_step * scale * np.array([ux, uy])
                self._clamp_i(i)
                self.phase[i] = 0  #reset cycle

        return self.positions.copy()

    def run(self):
        for t in range(1, self.total_time + 1):
            self.step()
            self.trajectory[t] = self.positions 
        return self.trajectory

#plotting functions 
def plot_concentration_Gaussian(bounds=(-50, 50), res = 200, x0=0.0,\
                                 y0=0.0, A=1.0, sx=20.0, sy=20.0):
    lo, hi = bounds 
    xs = np.linspace(lo, hi, res)
    ys = np.linspace(lo, hi, res)
    XX, YY = np.meshgrid(xs, ys, indexing = 'xy')
    Z = C_gaussian(XX, YY, x0, y0, A, sx, sy)
    plt.figure(figsize=(8,10))
    plt.imshow(Z, extent=[lo, hi, lo, hi], origin= 'lower', aspect = 'equal')
    plt.colorbar(label= "Concentration")
    plt.title("Gaussian Concentration")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_histograms(trajectory, times, bounds=(-50, 50)):
    for t in times:
        X_t = trajectory[t, :, 0] 
        plt.figure(figsize = (8, 5))
        plt.hist(X_t, bins= 40, range= bounds)
        plt.title(f"x_hist at t={t}")
        plt.xlabel("x")
        plt.ylabel("count")
        plt.show()

def plot_trajectories(trajectory, K= 100, bounds= (-50, 50)):
    T, N, _ = trajectory.shape
    K = min(K, N)
    idx= np.linspace(0, N - 1, K, dtype= int)
    plt.figure(figsize=(6,6))
    for i in idx:
        plt.plot(trajectory[:, i, 0], trajectory[:, i, 1], linewidth= 0.7, alpha= 0.7)
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectories (showing {K}/{N})")
    plt.show()

#main execution
if __name__ == "__main__":
    sim= Sim_EColi(N=10, total_time= 500, bounds=(-50,50))
    traj = sim.run()
    plot_concentration_Gaussian(bounds=(-50,50))
    plot_histograms(traj, times= [0, 50, 100, 150])
    plot_trajectories(traj, K= 20, bounds=(-50,50))
