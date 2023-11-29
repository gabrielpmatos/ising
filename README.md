# Ising Model Simulation
## Dependencies
* You should have some version of `python 3.x` with `numpy`, `matplotlib`, `numba`, and `multiprocessing` installed.
  * If you don't have these, you can always `pip install --user package_name` or add them to a `conda` environment.
  
## Simulation
* To simulate, run `python simulate.py`.
  * The default simulation will run for `L = [10, 16, 24, 36]` and 300 temperature points between 0.015 and 4.5.
  * The computationally heavy functions (i.e. `energy()`, `magnetization()`, and `mc_step_metropolis()` are all either vectorized or compiled just-in-time with numba.
  * The code should also detect how many CPU cores your computer has available and parallelize the calculation for each temperatue.
  * Without parallelization, the entire calculation (on an older Nevis computer) takes **~8-9 hrs**. On my MacBook Pro with parallelization to 12 cores, I estimate it should take **~30-45 mins**.
 
* The simulation will store the results for each lattice size `L` as structured arrays on `.npy` files under `./data`.
  * These should contain data for `E`, `M`, `C`, `Chi`, and `T` that are easy to read-in and plot.

## Analysis
* To make plots, run `python analysis.py`.
  * This will generate plots under `./plots`.
  * These include plots of the energy, magnetization, heat capacity, susceptibility, critical temperatures, etc.

### If you want to run your own analysis
* You can read-in the saved structured arrays by just replacing `X` below with the desired system size:
```Python
import numpy as np

with open("./path_to_data/ising_2d_L_X.npy", "rb") as f:
  array = np.load(f)
```
* You can then access the data from these arrays as `T = array["T"]` or `Chi = array["Chi"]` and run your own scripts.

> [!NOTE]
> I still need to finish fleshing out the analysis and writing some documentation for it, so just use it with this in mind!

