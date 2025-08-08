import numpy as np 
import matplotlib.pyplot as plt 

# gradient descent along concentration gradient, N =10, 100, 1000..
# procedure - four subsequent time steps: random walk (tumble) 
# calculate concentration gradient from t vs t- 4 delta t
# move along gradient(run)
# repeat
# plot histogram of locations of E.coli for different tt 
# plot motion of E.coli in x-y plane for different t 
# visualize concentration profile 
# experiment with three concentration profiles

def C_gaussian(x,y, x0= 0.0, y0= 0.0, A= 1.0, sx= 20.0, sy= 20.0)
    X, Y = x-x0, y- y0
    return A* np.exp(-0.5 * ((X/sx)**2 +(Y/sy)**2))

class Sim_biased_random_walk_EColi:
    def __init__(self, N =100, total_time = 200, bounds= (-100, 100),\
                  step_size = 1.0, seed=None, x0= 0.0, y0= 0.0, A= 1.0, sx= 20, sy= 20.0):
        self.N = int(N)
        self.total_time = int(total_time)
        self.bounds = tuple(bounds) 
        self.step_size = float(step_size)
        self.sigma = float(sigma)
        self.lo, self.hi = self.bounds 

        #random number generation
        if seed is not None:
            np.random.seed(seed)

        #gaussian parameters
        self.x0, self.y0, self.A, self.sx, self.sy = x0, y0, A, sx, sy

        #initialize positions
        xs= np.random.uniform(self.lo, self.hi, size = self.N)
        ys= np.random.uniform(self.lo, self.hi, size = self.N)
        self.positions = np.column_stack([xs, ys]).astype(float)
        
        #last direction (unit vectors)
        angles = np.random.uniform(0, 2 * np.pi, size= self.N)      #generate random angles
        self.last_direction = np.column_stack([np.cos(angles), np.sin(angles)])   #ranom tumbles

        #phase: 0 to 3 = tumbles; 4 = decision/run 
        self.phase = np.zeros(self.N, dtype= int)    

        #initial concentration 
        self.concentration_mark = np.zeros(self.N, dtype= float )              

        #trajectory storage: (T+1, N,2)
        self.trajectory = np.zeros((total_time + 1, N, 2), dtype= float)
        self.trajectory[0] = self.positions

    #concentration field
    def concentration_gaussian(self, x, y):
        return C_gaussian(x,y, self.x0, self.y0, self.A, self.sx, self.sy)
    
    def _clamp(self, i):                                       #ensure ecoli stay in bounds
        self.positions[i, 0] = np.clip(self.positions[i, 0], self.lo, self.hi)
        self.positions[i, 1] = np.clip(self.positions[i,1], self.lo, self.hi)
    
    def random_walk_dir(self):                          #random unit vectors
        angle = np.random.uniform(0, 2 * np.pi)      
        return np.array([np.cos(angle), np.sin(angle)], dtype = float)

    def step(self):
        for i in range(self.N):
            if self.phase == 0:       #phase 0 at start
                self.concentration_mark[i] = self.concentration_gaussian(\
                                  self.positions[i,0], self.positions[i,1])
            if self.phase[i] < 4:                          #tumble, pick rand.new direction, move one unit
                self.last_direction[i] = self.random_walk_dir()
                self.positions[i] += self.step_size * self.last_direction[i]
                self._clamp_i(i)
                self.phase[i] += 1
            else:                                
                Current_C = self.concentration_gaussian[i,0], self.positions[i, 1]  #compare current concentration vs marked concentration,run
                if current_C - self.concentration_mark[i] > 0:       #run, keep going in last direction 
                    self.positions[i] += self.step_size * self.last_direction[i]
                else:                               #tumble again 
                    self.last_direction[i] = self.random_walk_dir()
                    self.positions[i] += self.step_size * self.last_direction[i]
                self.clamp_i(i)
                self.phase[i] = 0  #reset cycle
        return self.positions.copy()

def run(self):
    for t in range(1, self.total_time + 1):
        self.step()
        self.trajectory[t] = self.positions 
    return self.trajectory

#plotting
def plot_concentration_Gaussian(bounds=(-100, 100), res = 200, x0=0.0,\
                                 y0=0.0, A=1.0, sx=20.0, sy=20.0):
    lo, hi = bounds 
    xs = np.linespace(lo, hi, res)
    ys = np.linespace(lo, hi, res)
    XX, YY = np.meshgrid(xs, ys, indexing = 'xy')
    Z = C_gaussian(XX, YY, x0, y0, A, sx, sy)
    plt.figure(figsize=(8,10))
    plt.imshow(Z, extent=[lo, hi, lo, hi], origin= 'lower', aspect = 'equal')
    plt.title("Gaussian Concentration")
    plt.xlabel("x")
    plt.ylabel("y")

def plot_histograms(trajectory, times, bounds= (-100,100)):
    times = [ t for t in times if 0 <= t < trajectory.shape[0]]
    fig, axes = plt.subplots(1, len(times), figsize=(4*len(times),3), sharey=True)
    if len(times) == 1: axes = [axes]
    for ax, t in zip(axes, times):
        X_t= trajectory[t, :, 0] 
        ax.hist(X_t, bins= 40, range= bounds)
        ax.set_title(f"x_hist at t={t}")
        ax.set_xlabel("x")
        ax.set_ylabel("count")




     
        
        





    
    





