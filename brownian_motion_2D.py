from math import sqrt
from scipy.stats import norm
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel, axis, title

def brownian(x0, n, dt, delta, out=None):
    # x0: initial conditions
    # n: number of steps
    # dt: time step
    # delta: motion "speed"
    # out: optional output array

    # Convert initial conditions to numpy array
    x0 = np.asarray(x0)

    # Generate random increments
    # norm.rvs generates normally distributed random samples
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # Compute cumulative sum of increments
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)

    # Add initial conditions to path
    out += np.expand_dims(x0, axis=-1)

    return out

delta = 1   # Controls "speed" of Brownian motion
T = 10.0    # Total simulation time
N = 500     # Number of time steps
dt = T/N    # Time step size
m = 20      # Number of independent paths (only for 1D motion)


##For 2D Brownian Motion, 2D trajectory in x-y plane
# Initial values of x.
x = np.empty((2,N+1)) #x[0]= x-coordinate and x[1]= y-coordiante
x[:, 0] = 0.0 #set the first column of all rows to 0

brownian(x[:,0], N, dt, delta, out=x[:,1:])

# Plot the 2D trajectory.
plot(x[0],x[1])

# Mark the start and end points.
plot(x[0,0],x[1,0], 'go')
plot(x[0,-1], x[1,-1], 'ro')

# More plot decorations.
title('2D Brownian Motion')
xlabel('x', fontsize=16)
ylabel('y', fontsize=16)
axis('equal')
grid(True)
show()


##For 1D Brownian Motion, 1D path over time
# Create empty array for paths
""" x = np.empty((m,N+1)) #paths, time steps

# Set initial value for all paths
x[:, 0] = 50

# Generate Brownian paths
brownian(x[:,0], N, dt, delta, out=x[:,1:])  ## x starts as:
                                            # [50, 50, 50, ...]  (first column)

                                            # After brownian() call:
                                            # x will be filled with Brownian motion paths
                                            # First column remains 50, subsequent columns get path values

# Create time axis
t = np.linspace(0.0, N*dt, N+1)

# Plot each path
for k in range(m):
    plot(t, x[k])
xlabel('t', fontsize=16)
ylabel('x', fontsize=16)
grid(True)
show() """
