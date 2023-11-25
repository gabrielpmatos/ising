import numpy as np
import scipy
import time
from numba import njit

def timer(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"Starting {func_name}")
        
        t1 = time.perf_counter()
        output = func(*args, **kwargs)
        t2 = time.perf_counter()
        
        print(f"Total time for {func_name}: {t2 - t1:.3f} s\n")
        return output
    
    return wrapper

@njit
def _energy(state, L, J):
    """
    Does the same calculation as in IsingModel.v_energy, but here njit pre-compiles
    the Python code so that the loops runs basically at C speed.
    """
    energy = 0

    for i in range(L):
        for j in range(L):
            s = state[i,j]
            nn_sum = state[(i+1)%L, j] + state[i,(j+1)%L] + state[(i-1)%L, j] + state[i,(j-1)%L]
            energy += s *nn_sum

    return -J*(energy/4)

class IsingModel:

    def __init__(self, L, T, J=1):
        self.L = L                                              # Lattice size
        self.T = T                                              # Temperature
        self.J = J                                              # Neighbor interaction strength
        self._nn = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Nearest neighbor structure 

        self.state = self._get_initial_state()

    def _get_initial_state(self):
        """
        Returns lattice of size LxL with up spins (+1) and down spins (-1) selected randomly.
        """
        return 2*np.random.randint(2, size=(self.L, self.L))-1

    @property
    def total_spins(self):
        return (self.L)**2

    @property
    @timer
    def v_energy(self):
        """
        Vectorized energy calculation.

        Calculates energy from H = -J * Sum_{nn} S_i S_j. This is done in this funny way
        to make sure that things are vectorized and fast. See this stack overflow post:
        https://stackoverflow.com/a/35925698

        This goes through every spin (double starred) and sums its nearest neighbors (starred),
        then the contribution to the energy is the (**) spin times the nearest neighbor sum.

        [[ 1,    1*,  -1,  -1]
         [-1*,  1**,   1*, -1]
         [ 1,   -1*,  -1,   1]
         [ 1,    1,   -1,  -1]]

        Doing this overcounts the nearest neighbor sum by 4, so we divide by 1/4.
        """
        return - (1/4) * self.J * np.sum(self.state * scipy.ndimage.convolve(self.state, self._nn, mode="wrap"))

    @property
    @timer
    def energy(self):
        """
        Calculates energy from H = -J * Sum_{nn} S_i S_j. 
        
        This goes through every spin (double starred) and sums its nearest neighbors (starred),
        then the contribution to the energy is the (**) spin times the nearest neighbor sum.

        [[ 1,    1*,  -1,  -1]
         [-1*,  1**,   1*, -1]
         [ 1,   -1*,  -1,   1]
         [ 1,    1,   -1,  -1]]

        Doing this overcounts the nearest neighbor sum by 4, so we divide by 1/4. 
        """
        return _energy(self.state, self.L, self.J)

    @property
    def magnetization(self):
        ...

if __name__ == "__main__":
    ising = IsingModel(10000, 2)
    print(ising.state)
    print(ising.dummy_energy)
    print(ising.dummy_energy)
    print(ising.energy)