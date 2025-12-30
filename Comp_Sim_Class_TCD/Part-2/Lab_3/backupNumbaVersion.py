from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import time


@njit
def delta_energy_numba(spins, h, J, i, j, L):
    """Compute ΔE for a single site (i, j) using periodic boundaries."""
    spin = spins[i, j]

    ip = (i + 1) % L
    im = (i - 1) % L
    jp = (j + 1) % L
    jm = (j - 1) % L

    neighbors = spins[ip, j] + spins[im, j] + spins[i, jp] + spins[i, jm]
    return 2 * spin * (J * neighbors + h[i, j])
@njit
def metropolis_numba(spins, h, J, kT, num_swaps):
    """Numba-accelerated Metropolis sweep. Updates spins IN PLACE."""
    L = spins.shape[0]
    flips = 0

    for _ in range(num_swaps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        dE = delta_energy_numba(spins, h, J, i, j, L)

        if dE <= 0 or np.random.rand() < np.exp(-dE / kT):
            spins[i, j] = -spins[i, j]
            flips += 1

    return flips



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
# CLASS 2: IsingLattice - Optimized with NumPy backend
# ============================================================================
class IsingLattice:
    """2D lattice of spins with Ising model dynamics (optimized)"""
    
    def __init__(self, L, J=1.0, kT=2.27, h=0.0, initial_state='random'):
        """
        Initialize L x L lattice of spins
        
        Parameters:
        -----------
        L : int
            Lattice size
        J : float
            Coupling constant
        kT : float
            Temperature (in units of J/k_B)
        h : float
        initial_state : str or np.ndarray
            Initial spin configuration:
            - 'random': Random ±1 spins (default)
            - 'up': All spins +1 (ferromagnetic)
            - 'down': All spins -1 (ferromagnetic)
            - 'checkerboard': Alternating ±1 pattern (antiferromagnetic)
            - np.ndarray: Custom L×L configuration
        """
        self.L = L
        self.J = J
        self.kT = kT
        if np.isscalar(h):
            self.h = np.full((L, L), h, dtype=float)
        else:
            if h.shape != (L, L):
                raise ValueError(f"Magnetic field must have shape ({L}, {L})")
            self.h = h.astype(float)
 
        self.initial_state = initial_state  # Store for reset
        self._spin_array = self._initialize_spins(initial_state)
    
    def _initialize_spins(self, initial_state):
        """
        Initialize spins based on initial_state parameter
        
        Parameters:
        -----------
        initial_state : str or np.ndarray
            Type of initial configuration
            
        Returns:
        --------
        np.ndarray : L×L array of ±1 spins
        """
        # Check for custom array FIRST (before string comparisons)
        if isinstance(initial_state, np.ndarray):
            # Custom configuration provided
            if initial_state.shape != (self.L, self.L):
                raise ValueError(f"Custom initial_state must have shape ({self.L}, {self.L})")
            return initial_state.copy()
        
        # Now check string options
        if initial_state == 'random':
            return np.random.choice([-1, 1], size=(self.L, self.L))
        
        elif initial_state == 'up':
            return np.ones((self.L, self.L), dtype=int)
        
        elif initial_state == 'down':
            return -np.ones((self.L, self.L), dtype=int)
        
        elif initial_state == 'checkerboard':
            pattern = np.zeros((self.L, self.L), dtype=int)
            # Create checkerboard pattern
            pattern[::2, ::2] = 1    # Even rows, even columns: +1
            pattern[1::2, 1::2] = 1  # Odd rows, odd columns: +1
            pattern[::2, 1::2] = -1  # Even rows, odd columns: -1
            pattern[1::2, ::2] = -1  # Odd rows, even columns: -1
            return pattern
        
        else:
            raise ValueError(f"Unknown initial_state: {initial_state}. "
                        f"Use 'random', 'up', 'down', 'checkerboard', or provide np.ndarray")
    
    @property
    def spins(self):
        """
        Returns spins as 2D list of Spin objects (for OOP compatibility)
        Only created when accessed (lazy evaluation)
        """
        return [[Spin(value=self._spin_array[m, n]) 
                 for n in range(self.L)] 
                for m in range(self.L)]
    
    def get_spin_value(self, m, n):
        """Get value of spin at position (m, n)"""
        return self._spin_array[m, n]
    
    def flip_spin(self, m, n):
        """Flip spin at position (m, n)"""
        self._spin_array[m, n] *= -1
    
    def get_neighbors_sum(self, m, n):
        """Sum of four nearest neighbors (periodic boundaries)"""
        L = self.L
        return (self._spin_array[(m+1) % L, n] +
                self._spin_array[(m-1) % L, n] +
                self._spin_array[m, (n+1) % L] +
                self._spin_array[m, (n-1) % L])


    def set_temperature(self, kT):
        """
        Set the temperature for the simulation
        
        Parameters:
        -----------
        kT : float
            Temperature (in units of J/k_B)
        """
        self.kT = kT


    def metropolis(self, kT=None):
        """Call Numba-accelerated Metropolis sweep."""
        if kT is not None:
            self.kT = kT

        num_swaps = 1000 * self.L * self.L

        # Call Numba function (updates array in-place)
        flips = metropolis_numba(
            self._spin_array,
            self.h,
            self.J,
            self.kT,
            num_swaps
        )

        return flips

    
    def magnetization(self):
        """Calculate average magnetization (optimized)"""
        return abs(np.mean(self._spin_array))
    
    def visualize(self, kT=None, title=None):
        """
        Display spin configuration
        
        Parameters:
        -----------
        kT : float, optional
            Temperature for title (uses self.kT if not provided)
        title : str, optional
            Custom title for the plot
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(self._spin_array, cmap='bwr', interpolation='nearest', vmin=-1, vmax=1)
        
        if title:
            plt.title(title, fontsize=14, fontweight='bold')
        else:
            temp_to_show = kT if kT is not None else self.kT
            plt.title(f"kT = {temp_to_show:.2f}", fontsize=14, fontweight='bold')
        
        plt.colorbar(label='Spin', ticks=[-1, 1])
        plt.tight_layout()
        plt.show()
        
    def reset(self, initial_state=None):
        """
        Reset lattice to initial configuration
        
        Parameters:
        -----------
        initial_state : str or np.ndarray, optional
            If provided, use this configuration
            If None, use the original initial_state from __init__
        """
        if initial_state is None:
            initial_state = self.initial_state
        else:
            self.initial_state = initial_state  # Update stored initial state
        
        self._spin_array = self._initialize_spins(initial_state)


    def set_magnetic_field(self, h):
        self.h = h

    def metropolis_step(self):
        """
        Perform one Monte Carlo sweep (L²  spin flip attempts)
        """
        L = self.L
        
        for _ in range(L * L):
            m, n = np.random.randint(0, L, size=2)
            dE = self.delta_energy(m, n)
            
            if dE < 0 or np.random.rand() < np.exp(-dE / self.kT):
                self.flip_spin(m, n)
    
    def energy(self):
        """
        Calculate total energy: E = -J Σ⟨i,j⟩ s_i·s_j - h Σ_i s_i
        
        Returns:
        --------
        float : Total energy
        """
        L = self.L
        
        # Part 1: Interaction energy (neighbor coupling)
        # Only count each pair ONCE (right and down neighbors)
        lattice_energy = 0.0
        for m in range(L):
            for n in range(L):
                spin = self._spin_array[m, n]
                right_neighbor = self._spin_array[m, (n+1) % L]
                down_neighbor = self._spin_array[(m+1) % L, n]
                h_local = self.h[m, n]
                # Each pair counted once
                lattice_energy += spin * right_neighbor + spin * down_neighbor + h_local * spin
        
        lattice_energy *= -self.J
        
        
        return float(lattice_energy)

# ============================================================================
# MAIN PROGRAM - Demonstrating different initial states
# ============================================================================
if __name__ == "__main__":
    
    '''
    # Test different initial states
    print("Testing different initial states:\n")
    
    # 1. Random initial state (default)
    lattice_random = IsingLattice(L, initial_state='random')
    print(f"Random: Magnetization = {lattice_random.magnetization():.3f}")
    lattice_random.visualize(title="Initial State: Random")
    
    # 2. All spins up
    lattice_up = IsingLattice(L, initial_state='up')
    print(f"All Up: Magnetization = {lattice_up.magnetization():.3f}")
    lattice_up.visualize(title="Initial State: All Up")
    
    # 3. All spins down
    lattice_down = IsingLattice(L, initial_state='down')
    print(f"All Down: Magnetization = {lattice_down.magnetization():.3f}")
    lattice_down.visualize(title="Initial State: All Down")
    
    # 4. Checkerboard pattern
    lattice_check = IsingLattice(L, initial_state='checkerboard')
    print(f"Checkerboard: Magnetization = {lattice_check.magnetization():.3f}")
    lattice_check.visualize(title="Initial State: Checkerboard")
    
    # 5. Custom initial state (half up, half down)
    custom_config = np.ones((L, L), dtype=int)
    custom_config[:, L//2:] = -1  # Right half is down
    lattice_custom = IsingLattice(L, initial_state=custom_config)
    print(f"Custom (half-half): Magnetization = {lattice_custom.magnetization():.3f}")
    lattice_custom.visualize(title="Initial State: Custom (Half-Half)")
    
    print("\n" + "="*60)
    print("Running temperature sweep with 'up' initial state:")
    print("="*60)
    '''

    L = 10

    
    # Temperature sweep starting from all spins up
    lattice = IsingLattice(L, initial_state='random', kT=1.5, h=1.0)
    temp = np.array([1.5, 1.8, 2.1, 2.2, 2.27, 2.4, 2.5, 2.7, 3.0, 3.5])
    magnetization = []
    
    start_time = time.time()
    
    for kT in temp:
        lattice.reset('up')  # Reset to all up for each temperature
        lattice.metropolis(kT)
        mag = lattice.magnetization()
        magnetization.append(mag)
        print(f"kT = {kT:.2f}: M = {mag:.3f}")
        #lattice.visualize(kT)
    
    end_time = time.time()
    print(f"\nSimulation time: {int(end_time - start_time)} seconds")