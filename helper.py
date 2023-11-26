import numpy as np
from numba import njit
import time

def timer(function):
    """
    Helper function to calculate total run time of functions
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
# Some njit functions, these are pre-compiled to make the calculations v. fast
#-----------------------------------------------------------------------------

@njit
def energy(state, L, J):
    """
    Actually does the energy calculation here. The njit decorator pre-compiles the 
    Python code so that the loops run basically at C speed.
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

    for _ in range(N):
        a = np.random.randint(0, L)
        b = np.random.randint(0, L)

        s = state[a, b]
        nn_sum = state[(a+1)%L, b] + state[a,(b+1)%L] + state[(a-1)%L, b] + state[a,(b-1)%L]

        cost = 2*J*s*nn_sum

        if cost < 0: state[a, b] = -s
        elif np.random.rand() < np.exp(cost*(1/T)): state[a, b] = -s

    return state