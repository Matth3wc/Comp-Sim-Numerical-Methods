import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# CLASS 1: Spin - Represents a single spin
# ============================================================================
class Spin:
    """Represents a single spin with value +1 or -1"""
    
    def __init__(self, value=None):
        """Initialize spin with random or given value"""
        if value is None:
            self.value = np.random.choice([-1, 1])
        else:
            self.value = value
    
    def flip(self):
        """Flip the spin"""
        self.value *= -1
    
    def __repr__(self):
        return f"Spin({self.value})"


# ============================================================================
# CLASS 2: IsingLattice - Contains all spins and simulation logic
# ============================================================================
class IsingLattice:
    """2D lattice of spins with Ising model dynamics"""
    
    def __init__(self, L, J=1.0):
        """
        Initialize L x L lattice of spins
        
        Parameters:
        -----------
        L : int
            Lattice size
        J : float
            Coupling constant
        """
        self.L = L
        self.J = J
        self.spins = [[Spin() for _ in range(L)] for _ in range(L)]
    
    def get_spin_value(self, m, n):
        """Get value of spin at position (m, n)"""
        return self.spins[m][n].value
    
    def flip_spin(self, m, n):
        """Flip spin at position (m, n)"""
        self.spins[m][n].flip()
    
    def get_neighbors_sum(self, m, n):
        """Sum of four nearest neighbors (periodic boundaries)"""
        L = self.L
        return (self.get_spin_value((m+1) % L, n) +
                self.get_spin_value((m-1) % L, n) +
                self.get_spin_value(m, (n+1) % L) +
                self.get_spin_value(m, (n-1) % L))
    
    def delta_energy(self, m, n):
        """Energy change if spin at (m,n) is flipped"""
        spin = self.get_spin_value(m, n)
        neighbors = self.get_neighbors_sum(m, n)
        return 2 * self.J * spin * neighbors
    
    def metropolis(self, kT):
        """Run Metropolis algorithm at temperature kT"""
        L = self.L
        num_swaps = 1000 * L * L
        
        for _ in range(num_swaps):
            m, n = np.random.randint(0, L, size=2)
            dE = self.delta_energy(m, n)
            
            if dE < 0 or np.random.rand() < np.exp(-dE / kT):
                self.flip_spin(m, n)
    
    def magnetization(self):
        """Calculate average magnetization"""
        total = sum(self.get_spin_value(m, n) 
                   for m in range(self.L) 
                   for n in range(self.L))
        return abs(total / (self.L * self.L))
    
    def visualize(self, kT=None):
        """Display spin configuration"""
        # Convert to numpy array for plotting
        spin_array = np.array([[self.get_spin_value(m, n) 
                               for n in range(self.L)] 
                              for m in range(self.L)])
        
        plt.imshow(spin_array, cmap='bwr', interpolation='nearest')
        if kT is not None:
            plt.title(f"kT = {kT:.2f}")
        plt.show()


# ============================================================================
# MAIN PROGRAM - Same as procedural version
# ============================================================================
if __name__ == "__main__":
    
    L = 50
    temp = np.array([1.5, 1.8, 2.1, 2.2, 2.27, 2.4, 2.5, 2.7, 3.0, 3.5])
    magnetization = []
    
    # Create lattice
    lattice = IsingLattice(L)
    
    start_time = time.time()
    
    for kT in temp:
        lattice.metropolis(kT)
        mag = lattice.magnetization()
        magnetization.append(mag)
        
        lattice.visualize(kT)
    
    end_time = time.time()
    print(f"Simulation time: {int(end_time - start_time)} seconds")