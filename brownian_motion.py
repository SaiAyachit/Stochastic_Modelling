import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Standard Brownina motion has drift_mean=0 and volatility= 1
mu,sigma = 0.0,1.0
paths = 50 #rows, no. of independent random walk simulations
points = 1000 #columns, no. of time steps in each simulation

#creates a random number class generator with a fixed seed of 42
rng = np.random.default_rng(42)

#Draw the samples from a std Normal Gaussian distribution
#mu = 0 (mean), sigma = 1 (std deviation) of the Normal distribution
Z = rng.normal(mu, sigma, (paths, points)) #Z will be a matrix of random numbers, (paths * points) shaped


#Display the histogram of samples,along with the PDF
count, bins, _ = plt.hist(Z, 30, density=True) 
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()


#time interval over which the simulation occurs
interval = [0,1]

#equal step sizes (delta T = Ti+1 - Ti)
dt = (interval[1] - interval[0]) / (points-1)

# linspace creates an array of 1000 equally spaced points from 0 to 1, useful for plotting
t_axis = np.linspace(interval[0], interval[1], points)

#Brownian Motion formula from [Glasserman 2003, section 3.2 Information Structure]
W=np.zeros((paths,points))
for idx in range(points-1):
    real_idx = idx+1
    W[:,real_idx] = W[:,real_idx -1] + np.sqrt(dt) * Z[:,idx]  #Next position = Previous position + (√Δt × Random draw)
    #We're simultaneously updating ALL 50 paths at a single time point
    #Each path gets its own random increment from Z
    #This vectorized approach is much faster than looping through paths individually

fig,ax = plt.subplots(1,1,figsize=(12,8))
for path in range(paths):
    ax.plot(t_axis, W[path, :])
ax.set_title("Standard Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Position")
plt.show()