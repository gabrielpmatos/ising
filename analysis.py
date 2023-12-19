import numpy as np
import matplotlib.pyplot as plt

data_path = "./data/"
plots_path = "./plots/"
sizes_to_plot = [10, 16, 24, 36]

def critical_values(T, X):
    # We get some outliers near T = 0, so use this to ignore those
    peak_window = (T > 1.5) & (T < 3.5)

    # Find T of peak, peak value, and return
    T, X = T[peak_window], X[peak_window]
    peak = np.argmax(X)
    return T[peak], X[peak]

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
                    results[L]["E"] = results[L]["E"]/L # So we can measure E per spin

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

                ax.plot(T, X, linestyle="--", color=self.colors[i], label=f"L = {L}")
                #ax.scatter(T, X, marker=self.markers[i], color=self.colors[i], s=self.marker_size, label=f"L = {L}")
            
            ax.set(xlabel="Temperature [J]", ylabel=label)
            ax.grid()
            ax.legend()
            plt.tight_layout()
            for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_{variable}.{ext}")

            plt.cla()
            plt.clf()
            plt.close() 

    def plot_critical_temperature(self):
        fig, ax = plt.subplots()

        critical_temperatures = []
        inv_L = []
        accepted_Tc = 2.269185

        for _, L in enumerate(self.sizes):
            Tc, _ = critical_values(self.results[L]["T"], self.results[L]["Chi"])
            critical_temperatures.append(Tc); inv_L.append(1/L)

        ax.scatter(inv_L, critical_temperatures, color="k", s=self.marker_size)
        ax.axhline(accepted_Tc, color="r", ls="--", label=f"Accepted $T_C$ = {accepted_Tc}")

        ax.set(xlabel=r"$L^{-1}$", ylabel=r"$T_{c}(L)$", ylim=(2, 2.5))
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_critical_temperature.{ext}")

        plt.cla()
        plt.clf()
        plt.close() 
        
def main():
    plotter = Plotter(sizes_to_plot)
    plotter.plot_results()
    plotter.plot_critical_temperature()

if __name__ == "__main__":
    main()
