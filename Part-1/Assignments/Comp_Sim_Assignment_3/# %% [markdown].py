# %% [markdown]
# # Assignment 3: Numerical Solution of an Ordinary Differential Equation
# 
# This notebook explores numerical solutions to the ordinary differential equation:
# 
# $
# f(x, t) = \frac{dx}{dt} = (1 + t)x + 1 - 3t + t^2
# $
# 
# ### Objectives:
# 1. **Direction Field**: Visualize the direction field for $ t \in [0, 5] $ and $ x \in [-3, 3] $.
# 2. **Numerical Methods**: Solve the ODE using:
#    - Euler Method
#    - Improved Euler Method
#    - Runge-Kutta Method
# 3. **Comparison**: Analyze the behavior of the solutions with different step sizes ($ h = 0.04 $ and $ h = 0.02 $).
# 4. **Critical Value**: Identify the starting value $ x(t=0) $ such that $ x(t=5) \in [-2.0, -1.9] $.
# 
# ### Deliverables:
# - Direction field visualization.
# - Numerical solutions plotted for all methods.
# - Observations on the accuracy and benefits of different integration schemes.
# - Python code for solving the ODE and generating the results.

# %% [markdown]
# ### Task 1:
# 1. **Direction Field**: Visualize the direction field for $ t \in [0, 5] $ and $ x \in [-3, 3] $. With,
# 
#             
# $f(x, t) = \frac{dx}{dt} = (1 + t)x + 1 - 3t + t^2$
#             

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define the ODE function
def f(x, t):
    return (1 + t) * x + 1 - 3 * t + t**2

# %%
def direction_field():
    # Create grid
    t_values = np.linspace(0, 5, 25)
    x_values = np.linspace(-3, 3, 25)

    # Create meshgrid
    T, X = np.meshgrid(t_values, x_values)

    # Calculate slopes (dx/dt) at each grid point
    dT = np.ones_like(T)  # dt = 1 (unit step in t direction)
    dX = f(X, T)           # dx/dt from our ODE

    # Normalize arrows for better visualization
    # This makes all arrows the same length, only showing direction
    magnitude = np.sqrt(dT**2 + dX**2)
    dT_norm = dT / magnitude
    dX_norm = dX / magnitude

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot direction field using quiver
    Q = ax.quiver(T, X, dT_norm, dX_norm, 
                    magnitude, 
                    cmap='viridis',
                    alpha=0.7,
                    scale=25,
                    width=0.003,
                    headwidth=4,
                    headlength=5)

    # Add colorbar to show magnitude
    cbar = plt.colorbar(Q, ax=ax)
    cbar.set_label(r'Slope Magnitude $|\frac{dx}{dt}|$', rotation=270, labelpad=20, fontsize=11)

    # Labels and formatting
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title(r'Direction Field for $\frac{dx}{dt} = (1+t)x + 1 - 3t + t^2$', 
                    fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(-3, 3)


    plt.tight_layout()
    plt.savefig('direction_field.png', dpi=300, bbox_inches='tight')

direction_field()
plt.savefig('/Users/mattthew/Documents/GitHub/Comp-Sim-Numerical-Methods/Part-1/Assignments/Comp_Sim_Assignment_3/plots/direction_field.pdf')
plt.show()


# %% [markdown]
# ### Task 2:
# 2. **Numerical Methods**: Solve the ODE using:
#    - Simple Euler Method
# 

# %%
# define simple euler update - From Blackboard scripts
def seuler(t, x, step):
    """Simple Euler method: x_new = x + h * f(x, t)"""
    x_new = x + step * f(x, t)
    return x_new

# %%


def ieuler(x, y, step):
    """
    Improved Euler where:
    x = time (t)
    y = position (x in assignment)
    """
    y_new = y + 0.5*step*( f(y, x) + f(y + step*f(y, x), x + step) )
    return y_new

# %%
def rk(t, x, h):
    k1 = f(x, t)
    k2 = f(x + 0.5*h*k1, t + 0.5*h)
    k3 = f(x + 0.5*h*k2, t + 0.5*h)
    k4 = f(x + h*k3, t + h)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# %%
# Initialize
t = 0.0
x = 0.0655
h = 0.04
t_end = 5.0
# Set a threshold for "infinity" (optional, for catching overflow before it happens)
x_threshold = 1e10  # Consider values above this as "infinity"


t_values_seular = [t]
x_values_seular = [x]

t_values_ieuler = [t]
x_values_ieuler = [x]

t_values_rk = [t]
x_values_rk = [x]


# %%
def integrate(h,xo=0.0655):
        # Initialize
    t = 0.0
    xo = 0.0655
    t_end = 5.0
    # Set a threshold for "infinity" (optional, for catching overflow before it happens)
    x_threshold = 1e10  # Consider values above this as "infinity"


    t_values_seular = [t]
    x_values_seular = [xo]

    t_values_ieuler = [t]
    x_values_ieuler = [xo]

    t_values_rk = [t]
    x_values_rk = [xo]

    x=xo

    # Iterate
    while t < t_end:
        # Check for infinity or NaN
        if np.isinf(x) or np.isnan(x) or abs(x) > x_threshold:
            print(f"Solution with simple eular method diverged at t = {t:.4f}, x = {x}")
            print("Terminating integration early.")
            break
        x = seuler(t, x, h)  # Update x
        t = t + h             # Update t
        t_values_seular.append(t)
        x_values_seular.append(x)

    t=0
    x = 0.0655 #Reset x value

    # Iterate
    while t < t_end:
        # Check for infinity or NaN
        if np.isinf(x) or np.isnan(x) or abs(x) > x_threshold:
            print(f"Solution with improved eular method diverged at t = {t:.4f}, x = {x}")
            print("Terminating integration early.")
            break

        x = ieuler(t, x, h)  # Update x
        t = t + h             # Update t


        t_values_ieuler.append(t)
        x_values_ieuler.append(x)

    t=0
    x = 0.0655 #Reset x value

    # Iterate
    while t < t_end:
        # Check for infinity or NaN
        if np.isinf(x) or np.isnan(x) or abs(x) > x_threshold:
            print(f"Solution with Runge Kutte diverged at t = {t:.4f}, x = {x}")
            print("Terminating integration early.")
            break
        x = rk(t, x, h)  # Update x
        t = t + h             # Update t

        
        t_values_rk.append(t)
        x_values_rk.append(x)
    return t_values_seular, x_values_seular, t_values_ieuler, x_values_ieuler, t_values_rk, x_values_rk


for step_size in [0.04, 0.02, 0.004, 0.0004, 0.0002]: 
    t_values_seular, x_values_seular, t_values_ieuler, x_values_ieuler, t_values_rk, x_values_rk = integrate(step_size)
    direction_field()


    plt.plot(t_values_seular, x_values_seular, label='Simple Euler', color='red')
    plt.plot(t_values_ieuler, x_values_ieuler, label='Improved Euler', color='blue')
    plt.plot(t_values_rk, x_values_rk, label='Runge Kutta 4', color='green')
    plt.legend(loc='upper left')
    plt.savefig('/Users/mattthew/Documents/GitHub/Comp-Sim-Numerical-Methods/Part-1/Assignments/Comp_Sim_Assignment_3/plots/direction_field_solution.pdf')
    plt.show()

# %%
# Part 4: Find x0 such that x(5) is in [-2.0, -1.9]

def solve_to_t5(x0, h=0.04):
    """Integrate from t=0 to t=5 using RK4, return x(5)"""
    t, x = 0.0, x0
    while t < 5.0:
        x = rk(t, x, h)
        t += h
    return x

# Bisection search
h = 0.04
target_min, target_max = -2.0, -1.9
target_mid = -1.95

# Initial bracket (adjust if needed)
x0_low, x0_high = 0.06, 0.072

# Bisection loop
for iteration in range(60):
    x0_mid = (x0_low + x0_high) / 2
    x5 = solve_to_t5(x0_mid, h)
    
    if x5 < target_mid:
        x0_low = x0_mid
    else:
        x0_high = x0_mid
    
    # Optional: print progress
    if iteration % 10 == 0:
        print(f"Iter {iteration}: x0 = {x0_mid:.8f}, x(5) = {x5:.6f}")

# Final result
x0_final = (x0_low + x0_high) / 2
x5_final = solve_to_t5(x0_final, h)

print(f"\nFinal result:")
print(f"x0 = {x0_final:.10f}")
print(f"x(5) = {x5_final:.6f}")
print(f"In target range? {target_min <= x5_final <= target_max}")

# %%
# Part 4: Graph the critical trajectory

# Find the critical x0
x0_critical = 0.0659226962
h = 0.04

# Integrate and store the trajectory
t = 0.0
x = x0_critical
t_values = [t]
x_values = [x]

while t < 5.0:
    x = rk(t, x, h)
    t += h
    t_values.append(t)
    x_values.append(x)

# Plot direction field with trajectory
direction_field()
plt.plot(t_values, x_values, 'r-', linewidth=2.5, label=f'x₀ = {x0_critical:.4f}')
plt.plot(t_values[0], x_values[0], 'go', markersize=8, label='Start')
plt.plot(t_values[-1], x_values[-1], 'ro', markersize=8, label=f'End: x(5) = {x_values[-1]:.2f}')
plt.legend(loc='upper left')
plt.savefig('critical_trajectory.pdf')
plt.show()

print(f"Starting point: t=0, x={x_values[0]:.6f}")
print(f"Ending point: t=5, x={x_values[-1]:.6f}")


