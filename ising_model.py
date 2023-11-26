import numpy as np
import helper

class IsingModel:

    def __init__(self, L, T, J=1, option="metropolis"):
        self.L = L      # Lattice size
        self.T = T*J    # Temperature (in units of J)
        self.J = J      # Neighbor interaction strength

        # Do MC step either w/ metropolis or cluster algorithms
        self.option = self._set_option(option)

        self.state = self._get_initial_state()

    def _set_option(self, option):
        option = option.lower()

        if option not in ["metropolis", "cluster"]:
            print(f"MC step option {option} not supported, defaulting to metropolis algorithm...")
            return "metropolis"
        
        else: return option

    def _get_initial_state(self):
        """
        Returns lattice of size LxL with up spins (+1) and down spins (-1) selected randomly.
        """
        return 2*np.random.randint(2, size=(self.L, self.L))-1

    @property
    def total_spins(self):
        return (self.L)**2

    @property
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
        return helper.energy(self.state, self.L, self.J)

    @property
    def magnetization(self):
        """
        Caclulates M = <S_i> = |Sum_{i} S_i| / N.

        Here 'i' goes through the entire lattice, and N is the total number of spins.
        """
        return np.abs(np.sum(self.state))/self.total_spins
    
    def mc_step(self):
        if self.option == "metropolis":
            helper.mc_step_metropolis(self.state, self.L, self.T, self.J, self.total_spins)

        else:
            print("Come back later!")
            return