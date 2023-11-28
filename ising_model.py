import numpy as np
import helper

class IsingModel:

    def __init__(self, L, T, J=1):
        """
        Class representing the Ising model for a given lattice size, temperature, and
        interaction term, with an option to update the spin configuration using the
        Metropolis algorithm

        Parameters
        ----------
        L : int
            System size.

        T : float
            System temperature.

        J : float
            Nearest neighbor interaction constant.
        
        """
        self.L = L      # Lattice size
        self.T = T*J    # Temperature (in units of J)
        self.J = J      # Neighbor interaction strength

        self.state = self._get_initial_state()

    def _get_initial_state(self):
        """
        Returns lattice of size LxL with up spins (+1) and down spins (-1). If the temperature
        is low enough, start with all spins up. This makes the thermalization step more likely
        to converge. Otherwise, start with a random configuration.
        """
        if self.T / self.J < 1 : return np.ones((self.L, self.L))
        else: return 2*np.random.randint(2, size=(self.L, self.L))-1

    @property
    def total_spins(self):
        """
        Returns the total number of spins N = L*L.
        """
        return (self.L)**2

    @property
    def energy(self):
        """
        Returns the total energy for the current spin configuration.
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
        """
        Performs a Monte Carlo (MC) step with the Metropolis algorithm. 
        """
        helper.mc_step_metropolis(self.state, self.L, self.T, self.J, self.total_spins) #NOTE this is pass by reference