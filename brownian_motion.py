import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Set random seed
rng = np.random.default_rng(42)

paths = 50
points = 1000

#Standard Brownina motion has drift_mean=0 and volatility= 1
mu,sigma = 0.0,1.0

#draw the samples from a std normal Gaussian distribution
# paths = rows , points = columns
Z = rng.normal(mu, sigma, (paths, points))

#time interval over which the simulation occurs
interval = [0,1]

#equal step sizes (delta T = Ti+1 - Ti)
dt = (interval[1] - interval[0] / (points-1))

#convert to linear space to plot
t_axis = np.linspace(interval[0], interval[1], points)

#Brownina Motion formula from [Glasserman 2003, section 3.2 Information Structure]
W=np.zeros((paths,points))
for idx in range(points-1):
    real_idx = idx+1
    W[:,real_idx] = W[:,real_idx -1] + np.sqrt(dt) * Z[:,idx]

fig,ax = plt.subplots(1,1,figsize=(12,8))
for path in range(paths):
    ax.plot(t_axis, W[path, :])
ax.set_title("Standard Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.show()