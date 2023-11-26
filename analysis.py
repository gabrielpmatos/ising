import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

data_path = "./data/"
plots_path = "./plots/"
sizes_to_plot = [10, 16, 24, 36]

class Plotter:

    def __init__(self, sizes_to_plot):
        self.sizes, self.results = self._read_results(sizes_to_plot)

        # Cosmetic parameters
        self.markers = ["D", "^", "v", "o"]
        self.colors = ["g", "r", "b", "k"]
        self.marker_size = 20

        # Variables and labels
        self.variables = ["E", "M", "C", "Chi"]
        self.labels = ["Energy", "Magnetization", "Heat Capacity", "Susceptibility"]

    def _read_results(self, sizes_to_plot):
        sizes = []
        results = {}

        for L in sizes_to_plot:
            try: 
                with open(data_path + f'ising_2d_L_{L}.npy', 'rb') as f:
                    results[L] = np.load(f)

                sizes.append(L)

            except FileNotFoundError:
                print(f"No results available for L = {L}, skipping...")
                continue

        return sizes, results 

    def plot_results(self):
        for variable, label in zip(self.variables, self.labels):

            fig, ax = plt.subplots()

            for i, L in enumerate(self.sizes):
                T = self.results[L]["T"]
                X = self.results[L][variable]

                ax.scatter(T, X, marker=self.markers[i], color=self.colors[i], s=self.marker_size, label=f"L = {L}")
            
            ax.set(xlabel="Temperature [J]", ylabel=label)
            ax.grid()
            ax.legend()
            plt.tight_layout()
            for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_{variable}.{ext}")

            plt.cla()
            plt.clf()
            plt.close() 


def main():
    plotter = Plotter(sizes_to_plot)
    plotter.plot_results()

if __name__ == "__main__":
    main()
