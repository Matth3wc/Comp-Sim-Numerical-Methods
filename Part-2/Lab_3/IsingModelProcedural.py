import numpy as np
import matplotlib.pyplot as plt
import time

#----------------------function definitions------------------------------------
def init_spins(L):
    # allocate random spin orientations [-1,+1] to LxL lattice
    spins = np.random.choice([-1, 1],size=(L, L))
    return spins

def delta_energy(spins, m, n):
    # calculate interaction energy of a spin at (m,n) with neighboring spins
    L = spins.shape[0]
    J = 1
    spin = spins[m, n]
    neighbors = spins[(m+1)%L, n] + spins[(m-1)%L, n] + spins[m, (n+1)%L] + spins[m, (n-1)%L]
    return 2 * J * spin * neighbors

def metropolis(spins, kT):
    # randomly select spin and calculate the energy associated with the spin
    # use metropolis criterion at temp=kT to decide whether to flip spin
    # repeat for numSwaps=1000*L**2
    L = spins.shape[0]
    numSwaps = 1000*L**2
    for _ in range(numSwaps):
        m, n = np.random.randint(0, L, size=2)
        dE = delta_energy(spins, m, n)
        if dE < 0 or np.random.rand() < np.exp(-dE / kT):
            spins[m, n] *= -1
    return spins

def measure(spins):
    # magnetiztion is the average spin value
    return abs(np.mean(spins))

#-----------------------------main program-------------------------------------
L = 50  
temp = np.array([1.5, 1.8, 2.1, 2.2, 2.27, 2.4, 2.5, 2.7, 3.0, 3.5])
magnetization = []

spins = init_spins(L)

start_time = time.time()

for kT in temp:
    spins = metropolis(spins, kT)
    mag = measure(spins)
    magnetization.append(mag)
    
    plt.imshow(spins,cmap='bwr',interpolation='nearest')
    plt.title(f"kT={kT:,.2f}")
    plt.show()
    
end_time = time.time()
print(f"Simulation time: {int(end_time - start_time)} seconds")
