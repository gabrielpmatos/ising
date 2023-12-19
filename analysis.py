import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data_path = "./data/"
plots_path = "./plots/"
sizes_to_plot = [10, 16, 24, 36]

# Helpers ================================

def critical_values(T, X):
    # We get some outliers near T = 0, so use this to ignore those
    peak_window = (T > 1.5) & (T < 3.5)

    # Find T of peak, peak value, and return
    T, X = T[peak_window], X[peak_window]
    peak = np.argmax(X)
    return T[peak], X[peak]

def fit_line(xx, yy):
    try:
        guess_m = (yy[1]-yy[0])/(xx[1]-xx[0])
        guess_b = 0
        guess = np.array([guess_m, guess_b])

        lin = lambda x, m, b: m*x + b

        popt, pcov = optimize.curve_fit(lin, xx, yy, p0 = guess)
        m, b = popt
        dm, db = pcov[0][0], pcov[1][1]
        fit_func = lin(xx, m, b)

    except IndexError:
        guess_m = 0
        m, b, dm, db, fit_func = 0, 0, 0, 0, 0
        print("!! Fit set to zeros !!")

    return(m, b, dm, db, fit_func)

# See https://stackoverflow.com/a/28766902
def find_intersection(x, y1, y2):
    idx = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    return x[idx], y1[idx], y2[idx]

def numerical_integrate(f, dx):
    return np.cumsum(f*dx)

# =========================================

class Plotter:

    def __init__(self, sizes_to_plot):
        self.sizes, self.results = self._read_results(sizes_to_plot)

        # Cosmetic parameters
        self.markers = ["D", "^", "v", "o"]
        self.colors = ["g", "r", "b", "k"]
        self.marker_size = 20

        # Variables and labels
        self.variables = ["E", "M", "C", "Chi"]
        self.labels = [r"$E/N$", "$M$", "$C_V$", "$\chi$"]

    def _read_results(self, sizes_to_plot):
        sizes = []
        results = {}

        for L in sizes_to_plot:
            try: 
                with open(data_path + f'ising_2d_L_{L}.npy', 'rb') as f:
                    results[L] = np.load(f)
                    results[L]["E"] /= L**2    # So we can measure E per spin

                sizes.append(L)

            except FileNotFoundError:
                print(f"No results available for L = {L}, skipping...")
                continue


        return sizes, results 

    # Answers Q3, parts of Q4, Q6, and Q7
    def plot_results(self):
        for variable, label in zip(self.variables, self.labels):

            fig, ax = plt.subplots()

            for i, L in enumerate(self.sizes):
                T = self.results[L]["T"]
                X = self.results[L][variable]

                ax.plot(T, X, linestyle="--", color=self.colors[i], label=f"L = {L}")
                if variable in ["C", "Chi"]:
                    Tc, Xc = critical_values(T, X)
                    ax.text(0.1, 0.8 - i*0.1, f"{label}$_c$ = {round(Xc,3)}, $T_c$ = {round(Tc,3)}", color=self.colors[i], transform=ax.transAxes)
             
            ax.set(xlabel="Temperature [J]", ylabel=label)
            ax.grid()
            ax.legend()
            plt.tight_layout()
            for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_{variable}.{ext}")

            plt.cla()
            plt.clf()
            plt.close() 

    # Answers Q4
    def plot_critical_temperature_chi(self):
        fig, ax = plt.subplots()

        critical_temperatures = []
        inv_L = []

        for i, L in enumerate(self.sizes):
            Tc, _ = critical_values(self.results[L]["T"], self.results[L]["Chi"])
            critical_temperatures.append(Tc); inv_L.append(1/L)
            ax.scatter(1/L, Tc, color=self.colors[i], s=self.marker_size, marker=self.markers[i], label=f"L = {L}")

        critical_temperatures, inv_L = np.array(critical_temperatures), np.array(inv_L)

        m, b, _, _, fit_func = fit_line(inv_L, critical_temperatures)
        ax.plot(inv_L, fit_func, color="orange", ls="--", label="Fit")
        ax.text(0.6, 0.3, r"$T_c(L) = T_c + (x_0 T_c)L^{-1}$", transform=ax.transAxes)
        ax.text(0.6, 0.2, f"$T_c$ = {round(b,3)}, $x_0 T_c$ = {round(m,3)}", transform=ax.transAxes)

        ax.set(xlabel=r"$L^{-1}$", ylabel=r"$T_{c}(L)$", ylim=(2.3, 2.45))
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_critical_temperature_chi.{ext}")

        plt.cla()
        plt.clf()
        plt.close()

    # Answers Q5
    def plot_gamma_nu(self):
        fig, ax = plt.subplots()

        log_Xc = []
        log_L = []
        
        for i, L in enumerate(self.sizes):
            _, Xc = critical_values(self.results[L]["T"], self.results[L]["Chi"])
            log_Xc.append(np.log(Xc)); log_L.append(np.log(L))
            ax.scatter(np.log(L), np.log(Xc), color=self.colors[i], s=self.marker_size, marker=self.markers[i], label=f"L = {L}") 

        log_Xc, log_L = np.array(log_Xc), np.array(log_L)

        m, b, _, _, fit_func = fit_line(log_L, log_Xc)
        ax.plot(log_L, fit_func, color="orange", ls="--", label="Fit")
        ax.text(0.6, 0.3, r"$\ln(\chi_c) = \frac{\gamma}{\nu}\ln(L) + f(0)$", transform=ax.transAxes)
        ax.text(0.6, 0.2, r"$\frac{\gamma}{\nu}$" + f" = {round(m,3)}, $f(0)$ = {round(b,3)}", transform=ax.transAxes)

        ax.set(xlabel=r"$\ln(L)$", ylabel=r"$\ln(\chi_c)$")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_gamma_nu.{ext}")

        plt.cla()
        plt.clf()
        plt.close()

    #Answers Q5
    def plot_chi_vs_scaling(self):

        log_Xc = []
        log_L = []
        
        for i, L in enumerate(self.sizes):
            _, Xc = critical_values(self.results[L]["T"], self.results[L]["Chi"])
            log_Xc.append(np.log(Xc)); log_L.append(np.log(L))

        log_Xc, log_L = np.array(log_Xc), np.array(log_L)
        gamma_nu, _, _, _, _ = fit_line(log_L, log_Xc)

        fig, ax = plt.subplots()

        for i, L in enumerate(self.sizes):
            T = self.results[L]["T"]
            X = self.results[L]["Chi"]
            Tc, _ = critical_values(self.results[L]["T"], self.results[L]["Chi"])

            ax.plot(L*(T-Tc), L**(-gamma_nu)*X, linestyle="--", color=self.colors[i], label=f"L = {L}") 

        ax.set(xlabel=r"$L^{1/\nu}(T-T_c(L))$", ylabel=r"$L^{-\gamma/\nu}\chi$")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_chi_vs_scaling.{ext}")

        plt.cla()
        plt.clf()
        plt.close()

    # Answers Q6
    def plot_beta_nu(self):

        beta_nu = 0.25

        fig, ax = plt.subplots()

        for i, L in enumerate(self.sizes):
            T = self.results[L]["T"]
            M = self.results[L]["M"]

            ax.plot(T, L**(beta_nu)*M, linestyle="--", color=self.colors[i], label=f"L = {L}") 
        
        Tc, Mc, _ = find_intersection(T, self.sizes[0]**(beta_nu)*self.results[self.sizes[0]]["M"], 
                                        self.sizes[-1]**(beta_nu)*self.results[self.sizes[-1]]["M"])
        
        ax.plot(Tc, Mc, color="orange", marker="*", markersize=12, linewidth=0, label = f"$T_c$ = {round(Tc[0], 3)}")

        ax.set(xlabel="Temperature [J]", ylabel=r"$L^{\beta/\nu}M$")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_beta_nu.{ext}")

        plt.cla()
        plt.clf()
        plt.close()

    # Answers Q6
    def plot_M_vs_scaling(self):

        beta_nu = 0.25
        Tc, _, _ = find_intersection(self.results[self.sizes[0]]["T"], 
                                     self.sizes[0]**(beta_nu)*self.results[self.sizes[0]]["M"], 
                                     self.sizes[-1]**(beta_nu)*self.results[self.sizes[-1]]["M"])
        
        fig, ax = plt.subplots()

        for i, L in enumerate(self.sizes):
            T = self.results[L]["T"]
            M = self.results[L]["M"]
            ax.plot(L*(T-Tc), L**(beta_nu)*M, linestyle="--", color=self.colors[i], label=f"L = {L}") 

        ax.set(xlabel=r"$L^{1/\nu}(T-T_c(L))$", ylabel=r"$L^{\beta/\nu}M$")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_M_vs_scaling.{ext}")

        plt.cla()
        plt.clf()
        plt.close()

    # Answers Q7
    def plot_critical_temperature_C(self):
        fig, ax = plt.subplots()

        critical_temperatures = []
        inv_L = []

        for i, L in enumerate(self.sizes):
            Tc, _ = critical_values(self.results[L]["T"], self.results[L]["C"])
            critical_temperatures.append(Tc); inv_L.append(1/L)
            ax.scatter(1/L, Tc, color=self.colors[i], s=self.marker_size, marker=self.markers[i], label=f"L = {L}")

        critical_temperatures, inv_L = np.array(critical_temperatures), np.array(inv_L)

        m, b, _, _, fit_func = fit_line(inv_L, critical_temperatures)
        ax.plot(inv_L, fit_func, color="orange", ls="--", label="Fit")
        ax.text(0.6, 0.3, r"$T_c'(L) = T_c' + (x_0 T_c')L^{-1}$", transform=ax.transAxes)
        ax.text(0.6, 0.2, f"$T_c'$ = {round(b,3)}, $x_0 T_c'$ = {round(m,3)}", transform=ax.transAxes)

        ax.set(xlabel=r"$L^{-1}$", ylabel=r"$T_{c}'(L)$", ylim=(2.25, 2.4))
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_critical_temperature_C.{ext}")

        plt.cla()
        plt.clf()
        plt.close()

    # Answers Q7
    def plot_C_divergence(self):
        fig, ax = plt.subplots()

        critical_C = []
        log_L = []

        for i, L in enumerate(self.sizes):
            _, Cc = critical_values(self.results[L]["T"], self.results[L]["C"])
            critical_C.append(Cc); log_L.append(np.log(L))
            ax.scatter(np.log(L), Cc, color=self.colors[i], s=self.marker_size, marker=self.markers[i], label=f"L = {L}")

        critical_C, log_L = np.array(critical_C), np.array(log_L)
       
        ax.set(xlabel=r"$\ln(L)$", ylabel=r"$C_c(L)$")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_C_divergence.{ext}")

        plt.cla()
        plt.clf()
        plt.close() 

    # Answers Q8
    def plot_entropy(self):
        fig, ax = plt.subplots() 

        for i, L in enumerate(self.sizes):
            T = self.results[L]["T"]
            C = self.results[L]["C"]
            S = numerical_integrate(np.diff(T)[0], C/T)
            ax.plot(T, S, linestyle="--", color=self.colors[i], label=f"L = {L}")

        ax.axhline(y = np.log(2), color="orange", ls="--", label=r"$\ln(2)$")

        ax.set(xlabel="Temperature [J]", ylabel=r"$S(T)$", xlim=(T.min(), T.max()))
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_entropy.{ext}")

        plt.cla()
        plt.clf()
        plt.close() 

    # Answers Q8
    def plot_free_energy(self):
        fig, ax = plt.subplots() 

        for i, L in enumerate(self.sizes):
            T = self.results[L]["T"]
            C = self.results[L]["C"]
            E = self.results[L]["E"]
            S = numerical_integrate(np.diff(T)[0], C/T)
            ax.plot(T, E-T*S, linestyle="--", color=self.colors[i], label=f"L = {L}")

        ax.set(xlabel="Temperature [J]", ylabel=r"$F = E-TS$")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]: fig.savefig(plots_path + f"ising_2d_free_energy.{ext}")

        plt.cla()
        plt.clf()
        plt.close() 

def main():
    plotter = Plotter(sizes_to_plot)
    plotter.plot_results()
    plotter.plot_critical_temperature_chi()
    plotter.plot_gamma_nu()
    plotter.plot_chi_vs_scaling()
    plotter.plot_beta_nu()
    plotter.plot_M_vs_scaling()
    plotter.plot_critical_temperature_C()
    plotter.plot_C_divergence()
    plotter.plot_entropy()
    plotter.plot_free_energy()

if __name__ == "__main__":
    main()
