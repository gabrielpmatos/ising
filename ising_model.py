import numpy as np
import helper

class IsingModel:

    def __init__(self, L, T, J=1, option="metropolis"):
        """
        Class representing the Ising model for a given lattice size, temperatue, and
        interaction term, with an option to update the spin configuration using either
        the Metropolis of cluster algorithms.

        Parameters
        ----------
        L : int
            System size.

        T : float
            System temperature.

        J : float
            Nearest neighbor interaction constant.

        option : string
            Update system in a Monte Carlo step using either the metropolis or cluster
            algorithms
        
        """
        self.L = L      # Lattice size
        self.T = T*J    # Temperature (in units of J)
        self.J = J      # Neighbor interaction strength

        # Do MC step either w/ metropolis or cluster algorithms
        self.option = self._set_option(option)

        self.state = self._get_initial_state()

    def _set_option(self, option):
        """
        Sets the MC option to either metropolis or cluster

        NOTE: the clustering option still needs to be implemented

        """
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
        Performs a Monte Carlo (MC) step. 

        If option is to 'metropolis', the metropolis algorithm is used. Otherwise,
        the cluster algorithm is used (this is useful when we are closer to the
        critical temperatue, and flipping individual spins becomes probabilistically
        unfavorable). 

        NOTE: the cluster option still needs to be implemented

        """
        if self.option == "metropolis":
            helper.mc_step_metropolis(self.state, self.L, self.T, self.J, self.total_spins)

        else:
            print("Come back later!")
            return