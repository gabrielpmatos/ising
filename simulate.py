from ising_model import IsingModel

def main():
    sweeps = 100000

    ising = IsingModel(36, 1)
    for i in range(sweeps):
        ising.mc_step()
        if i % 1000 == 0: print(ising.state)

if __name__ == "__main__":
    main()