import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from ising_model import IsingModel

data_path = "./data/"

def simulate(thermalize_sweeps, msmt_sweeps, msmt_rate, L, T, verbose=False):
    """
    Runs the simulation for the ising model.

    Parameters
    ----------
    thermalize_sweeps : int
        Number of sweeps to thermalize system.

    msmt_sweeps : int
        Number of sweeps done at measurement step.

    msmt_rate : int
        Rate at which measurements are made. I.e. if 10, a measurement is made
        every 10 measurement sweeps.

    L : int
        System size to simulate.

    T : float
        Temperature to simulate.

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
        ising.mc_step()

    if verbose: print("Done thermalizing!")

    for i in range(msmt_sweeps):
        ising.mc_step()

        if i % msmt_rate == 0:
            msmt_energy, msmt_magnetization = ising.energy, ising.magnetization

            energy += msmt_energy
            energy_sq += msmt_energy**2
            magnetization += msmt_magnetization
            magnetization_sq += msmt_magnetization**2

    if verbose: print(f"Done measurements for L = {L}, T = {T}!")

    return energy/n_msmts, energy_sq/n_msmts, magnetization/n_msmts, magnetization_sq/n_msmts

def save(results, L):
    """
    Saves simulation results to .npy files.

    Parameters
    ----------
    results : np.ndarray
        Values to be saved. This is a structured array.
    
    L : int
        Size of simulated system, used in the file name.
    
    """
    with open(data_path + f"ising_2d_L_{L}.npy", "wb") as f: np.save(f, results) 


#-----------------------------------------------------------------------------
def main():

    print("")
    print("~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/")
    print("")
    print(" _____    _              ___  ___          _      _ ")
    print("|_   _|  (_)             |  \/  |         | |    | |")
    print("  | | ___ _ _ __   __ _  | .  . | ___   __| | ___| |")
    print("  | |/ __| | '_ \ / _` | | |\/| |/ _ \ / _` |/ _ \ |")
    print(" _| |\__ \ | | | | (_| | | |  | | (_) | (_| |  __/ |")
    print(" \___/___/_|_| |_|\__, | \_|  |_/\___/ \__,_|\___|_|")
    print("                   __/ |                            ")
    print("                  |___/                             ")
    print("")
    print("~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/~/")
    print("")


    # Sweep parameters
    thermalize_sweeps = 10**5
    msmt_sweeps = 3*10**5
    msmt_rate = 10

    # System parameters
    sizes = [10, 16, 24, 36]
    T = np.arange(0.015, 4.5, 0.015) 
    k = 1 # Setting Boltzmann constant to 1

    print("Parameters")
    print("----------")
    print(f"Thermalization sweeps = {thermalize_sweeps}")
    print(f"Measurement sweeps    = {msmt_sweeps}")
    print(f"Measurement rate      = {msmt_rate}")
    print(f"System sizes          = {sizes}")
    print(f"Temperature range     = ({T.min()}, {T.max()}, {T.size})")
    print(f"Boltzmann constant    = {k}")
    print("")

    # Data type definitions for structured array of results
    results_dtype = [("E", float), ("M", float), ("C", float), ("Chi", float), ("T", float)]

    for L in sizes:
        print(f"Starting simulation for size L = {L}")
        N = L**2
        results = np.empty(T.size, dtype=results_dtype)   # Defines structured array to save

        # Parallelizes process depending on how many CPU cores are available
        with Pool(cpu_count()) as p:
            f = partial(simulate, thermalize_sweeps, msmt_sweeps, msmt_rate, L)
            energy, energy_sq, magnetization, magnetization_sq = map(np.asarray, zip(*p.map(f, T)))

        # Saves simulation results for analysis
        results["E"]   = energy
        results["M"]   = magnetization
        results["C"]   = (1/(N*k*T**2))*(energy_sq - energy**2)
        results["Chi"] = (N/(k*T))*(magnetization_sq - magnetization**2) 
        results["T"]   = T

        print(f"Done simulation for L = {L}!")
        print("Saving...")
        save(results, L)
        print("Done!")
        print("")


if __name__ == "__main__":
    main()