import argparse
import numpy as np
from ising_model import IsingModel

data_path = "./data/"

def simulate(L, T, thermalize_sweeps, msmt_sweeps, msmt_rate, verbose=False):
    """
    Runs the simulation for the ising model.

    Parameters
    ----------
    L : int
        System size to simulate.

    T : float
        Temperature to simulate.

    thermalize_sweeps : int
        Number of sweeps to thermalize system.

    msmt_sweeps : int
        Number of sweeps done at measurement step.

    msmt_rate : int
        Rate at which measurements are made. I.e. if 10, a measurement is made
        every 10 measurement sweeps.

    verbose : bool
        If 'True', turns on printouts for steps in sweep.

    Returns
    -------
    tuple(float)
        Returns average energy, energy squared, magnetization, and magnetization
        squared for a total (msmt_sweeps//msmt_rate) number of configurations
        post thermalization.
    
    """
    energy, energy_sq, magnetization, magnetization_sq = 0., 0., 0., 0.
    n_msmts = msmt_sweeps // msmt_rate

    ising = IsingModel(L, T)

    for i in range(thermalize_sweeps):
        if (i % 10000 == 0) and verbose: print(f"-----> Thermalization sweep #{i}")
        ising.mc_step()
    
    print("Done thermalizing!")

    for i in range(msmt_sweeps):
        if (i % 10000 == 0) and verbose: print(f"-----> Measurement sweep #{i}")
        ising.mc_step()

        if i % msmt_rate == 0:
            msmt_energy, msmt_magnetization = ising.energy, ising.magnetization

            energy += msmt_energy
            energy_sq += msmt_energy**2
            magnetization += msmt_magnetization
            magnetization_sq += msmt_magnetization**2

    print(f"Done measurements for L = {L}, T = {T}\n")
    return energy/n_msmts, energy_sq/n_msmts, magnetization/n_msmts, magnetization_sq/n_msmts

def save(values, variable, L=None):
    """
    Saves simulation results to .npy files.

    Parameters
    ----------
    values : list or np.ndarray
        Values to be saved.
    
    variable : string
        Name of variable to be saved.

    L : int
        Size of simulated system, used in the file name. If 'None' (as in 
        the case of saving the temperature array), the system size won't 
        be included in the file name.
    
    """
    if L: 
        with open(data_path + f"ising_2d_L_{L}_{variable}.npy", "wb") as f: np.save(f, values)
    else: 
        with open(data_path + f"ising_2d_{variable}.npy", "wb") as f: np.save(f, values) 

#-----------------------------------------------------------------------------
def main():

    # Sweep parameters
    thermalize_sweeps = 10**5
    msmt_sweeps = 3*10**5
    msmt_rate = 10

    # System parameters
    sizes = [10, 16, 24, 36]
    temperatures = np.arange(0.015, 4.5, 0.015) 
    k = 1 # Setting Boltzmann constant to 1

    for L in sizes:
        N = L**2
        E, M, C, Chi = [], [], [], []

        for T in temperatures:
            energy, energy_sq, magnetization, magnetization_sq = simulate(L, T, thermalize_sweeps, msmt_sweeps, msmt_rate)

            E.append(energy)
            M.append(magnetization)
            C.append((1/(N*k*T**2))*(energy_sq - energy**2))
            Chi.append((N/(k*T))*(magnetization_sq - magnetization**2))

        # Saves simulation results for analysis
        for values, variable in zip([E, M, C, Chi], ["E", "M", "C", "Chi"]): save(values, variable, L)

    # Saves array of temperature values
    save(temperatures, "T")

if __name__ == "__main__":
    main()