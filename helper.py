import numpy as np
from numba import njit
import time

def timer(function):
    """
    Helper function to calculate total run time of functions.
    """
    def wrapper(*args, **kwargs):
        function_name = function.__name__
        print(f"Starting {function_name}")
        
        t1 = time.perf_counter()
        output = function(*args, **kwargs)
        t2 = time.perf_counter()
        
        print(f"Total time for {function_name}: {t2 - t1:.3f} s")
        return output

    return wrapper

#-----------------------------------------------------------------------------
# Some njit functions, these are compiled during runtime to make the 
# calculations v. fast; below the loops work at essentially C speed.

# See the numba documentation: 
# https://numba.readthedocs.io/en/stable/user/5minguide.html
#-----------------------------------------------------------------------------

@njit
def energy(state, L, J):
    """
    Calculates energy from H = -J * Sum_{nn} S_i S_j. 
        
    This goes through every spin (double starred) and sums its nearest neighbors (starred),
    then the contribution to the energy is the (**) spin times the nearest neighbor sum.

    [[ 1,    1*,  -1,  -1]
     [-1*,  1**,   1*, -1]
     [ 1,   -1*,  -1,   1]
     [ 1,    1,   -1,  -1]]

    Doing this overcounts the nearest neighbor sum by 4, so we divide by 1/4. 

    Parameters
    ----------
    state : np.ndarray
        Current ising spin state.

    L : int
        System size being simulated.

    J : float
        Nearest neighbors interaction strength coefficient.

    Returns
    -------
    float
        The energy measurement for the given spin configuration.

    """
    energy = 0

    for i in range(L):
        for j in range(L):
            s = state[i,j]
            nn_sum = state[(i+1)%L, j] + state[i,(j+1)%L] + state[(i-1)%L, j] + state[i,(j-1)%L]
            energy += s *nn_sum

    return -J*(energy/4)

@njit
def mc_step_metropolis(state, L, T, J, N):
    """
    Does a Monte Carlo (MC) step using the Metropolis algorithm. 
    
    Loop through the number of spin sites of the system. For every iterations, 
    pick a random test spin site to flip. The energy cost of flipping a single 
    spin at location i, j is 2*J*s_{i,j}*Sum_{nn}, and if this energy cost is < 0, 
    we accept the spin flip. Otherwise, if the ratio of the P_new/P_old probabilities 
    is >= r (where r is a random number between 0 and 1), we keep the change. Else, 
    the spin flip is rejected. 

    Parameters
    ----------
    state : np.ndarray
        Current ising spin state.

    L : int
        System size being simulated.

    T : float
        Temperature of the system being simulated.

    J : float
        Nearest neighbors interaction strength coefficient.

    N : int
        Total number of spins in the system.

    Returns
    -------
    np.ndarray
        A new ising state with the accepted changes.
        
    """
    for _ in range(N):
        a = np.random.randint(0, L)
        b = np.random.randint(0, L)

        s = state[a, b]
        nn_sum = state[(a+1)%L, b] + state[a,(b+1)%L] + state[(a-1)%L, b] + state[a,(b-1)%L]

        cost = 2*J*s*nn_sum

        if cost < 0: state[a, b] = -s
        elif np.random.rand() <= np.exp(-cost*(1/T)): state[a, b] = -s

    return state