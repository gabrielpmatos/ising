# Ising Model Simulation
## Dependencies
* You should have some version of `python 3.x` with `numpy`, `matplotlib`, and `numba` installed.
  * If you don't have these, you can always `pip install --user package_name` or add them to a `conda` environment.
  
## Simulation
* To simulate, run `python simulate.py`.
  * You might want to try a smaller range of temperatures or just `L = 10` on a first pass, because the entire simulation takes quite a few (~9) hours to run for all 4 `L = 10, 16, 24, 36`.
  * To do so, just change the parameters in `main()` on `simulate.py`.
* The simulation will store the results for each lattice size `L` on `.npy` files under `./data`.

## Analysis
* To make plots, run `python analysis.py`.
  * This will generate plots under `./plots`.
  * These include plots of the energy, magnetization, heat capacity, susceptibility, critical temperature, etc.

### If you want to run your own analysis
* The results are stored as structured arrays containing data for `E`, `M`, `C`, `Chi`, and `T`. You can read-in these arrays with
```Python
import numpy as np

with open("ising_2d_L_10.npy", "rb") as f:
  array = np.load(f)
```
* You can then access the data from these arrays as `T = array["T"]` or `Chi = array["Chi"]` like a dictionary and run your own scripts.
