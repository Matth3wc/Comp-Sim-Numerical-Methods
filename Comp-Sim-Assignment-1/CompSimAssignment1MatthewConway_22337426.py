"""
PYU33C01 Assignment 1: Iterations and Numerical Accuracy
Student Name: Matthew Conway
Student Number: 22337426
Date: October 1, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

print("PYU33C01 Assignment 1")
print("Student Name: Matthew Conway")
print("Student Number: 22337426\n")

# ============================================================================
# SECTION A: HERON'S ROOT FINDING METHOD
# ============================================================================

print("SECTION A: HERON'S ROOT FINDING METHOD\n")

# Part 1: Implementation
def heron_root(a, x_0, n):
    x_n = x_0
    for _ in range(n):
        x_n = 0.5 * (x_n + a / x_n)
    return x_n

sqrt_2 = heron_root(2, 1, 10)

print(f"Approximation of x=√2 using Heron's method: {sqrt_2}")
print(f"NumPy sqrt(2): {np.sqrt(2)}")
print(f"Difference: {abs(sqrt_2 - np.sqrt(2))}")

# Part 2: Plot x_n as a function of iteration number
x_0_array = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])
n_array = np.arange(1, len(x_0_array)+1)

plt.figure(figsize=(8, 6)) 
for j in range(len(x_0_array)):

    heron_root_array = []
    markers = ['o', 's', '^', 'D', 'x', 'P', 'v', '>'] 

    x_0 = x_0_array[j]

    heron_root_array = np.array([heron_root(2, x_0, n-1) for n in n_array])

    line, = plt.plot(n_array, heron_root_array, linewidth=1, label=fr"$x_0$ = {x_0}")

    
    plt.plot(n_array, heron_root_array, linestyle='None',
             marker=markers[j],
             color=line.get_color(),)
        

plt.xlabel(f'Iteration number (n)')
plt.ylabel('Approximation of $\sqrt{2}$')
plt.title("Convergence of Heron's Method to $x=\sqrt{2}$")
plt.axhline(y=np.sqrt(2), color='black', linestyle='--', alpha=0.6, label='np.sqrt(2) result')
plt.xlim(1, 5)
plt.ylim(0, 2.5)
plt.legend()
plt.savefig('/Users/mattthew/Documents/GitHub/Comp-Sim-Numerical-Methods/Comp-Sim-Assignment-1/plots/herons_convergence.pdf',)
plt.show()


# Part 3: Plot relative error for x_n × x_n
def heron_square_error(a, x_0, n):
    x_n = heron_root(a, x_0, n)
    error = abs((x_n**2 - a) / a)
    return error

plt.figure(figsize=(8, 6)) 
for j in range(len(x_0_array)):

    heron_square_error_array = []
    markers = ['o', 's', '^', 'D', 'x', 'P', 'v', '>'] 

    x_0 = x_0_array[j]

    heron_square_error_array = np.array([heron_square_error(2, x_0, n) for n in n_array])

    line, = plt.plot(n_array, heron_square_error_array, linewidth=1, label=fr"$x_0$ = {x_0}")

    
    plt.plot(n_array, heron_square_error_array, linestyle='None',
             marker=markers[j],
             color=line.get_color(),)
        


plt.xlabel(f'Iteration number (n)')
plt.ylabel('Relative error $|x_n^2 - a|/a$')
plt.title("Relative error in Heron's Method for $\sqrt{2}$")
plt.yscale('log')
plt.xlim(1, 7)
plt.legend()
plt.savefig('/Users/mattthew/Documents/GitHub/Comp-Sim-Numerical-Methods/Comp-Sim-Assignment-1/plots/herons_error.pdf')
plt.show()


# ============================================================================
# SECTION B: NUMERICAL ACCURACY
# ============================================================================

print("SECTION B: NUMERICAL ACCURAC\n")

# Section 1.3: Dealing with Floating Point Numbers

# Question 1a: Underflow limit
print("Question 1a: Underflow Limit")

x = 1.0
underflow_limit = x
iteration = 0

while x / 2.0 > 0:
    underflow_limit = x
    x = x / 2.0
    iteration += 1

print(f"Underflow limit: {underflow_limit:.5e}")
print(f"Reached after {iteration} iterations")
print(f"System float info (min): {sys.float_info.min:.5e}\n")

# Question 1b: Overflow limit
print("Question 1b: Overflow Limit")

x = 1.0
overflow_limit = x
iteration = 0

while x * 2.0 < float('inf'):
    overflow_limit = x
    x = x * 2.0
    iteration += 1

print(f"Overflow limit: {overflow_limit:.5e}")
print(f"Reached after {iteration} iterations")
print(f"System float info (max): {sys.float_info.max:.5e}\n")

# Question 2: Machine precision
print("Question 2: Machine Precision (epsilon)")

eps = 1.0
machine_precision = eps
iteration = 0

while (1.0 + eps / 2.0) != 1.0:
    machine_precision = eps
    eps = eps / 2.0
    iteration += 1

print(f"Machine precision (epsilon): {machine_precision:.5e}")
print(f"Reached after {iteration} iterations")
print(f"System float info (epsilon): {sys.float_info.epsilon:.5e}\n")


# Section 1.4: Numerical Derivatives

def f(x):
    """Function to differentiate: f(x) = x(x-1)"""
    return x * (x - 1)

def analytical_derivative(x):
    """Analytical derivative: f'(x) = 2x - 1"""
    return 2 * x - 1

def forward_difference(f, x, h):
    """Forward difference approximation: [f(x+h) - f(x)]/h"""
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h):
    """Central difference approximation: [f(x+h) - f(x-h)]/(2h)"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Question 1a: Compare forward and central differences
print("Question 1a: Forward vs Central Difference")

x_point = 1.0
true_derivative = analytical_derivative(x_point)
h_values = np.logspace(-1, -16, 100)

forward_errors = []
central_errors = []

for h in h_values:
    forward_deriv = forward_difference(f, x_point, h)
    central_deriv = central_difference(f, x_point, h)
    
    forward_error = abs(forward_deriv - true_derivative)
    central_error = abs(central_deriv - true_derivative)
    
    forward_errors.append(forward_error)
    central_errors.append(central_error)

forward_errors = np.array(forward_errors)
central_errors = np.array(central_errors)

print(f"Function: f(x) = x(x-1) at x = {x_point}")
print(f"True derivative: f'({x_point}) = {true_derivative}")

# Plot the errors
plt.figure(figsize=(10, 6))
plt.loglog(h_values, forward_errors, 'b-', label='Forward Difference', linewidth=1)
plt.loglog(h_values, central_errors, 'r-', label='Central Difference', linewidth=1)
plt.axhline(y=machine_precision, color='g', linestyle='--', label=f'Machine Precision ({machine_precision:.2e})', linewidth=1.)

# Add reference lines for expected behavior
h_squared = h_values**2
plt.loglog(h_values, h_values * 1e-1, 'b:', alpha=0.5, label='$\propto h$ (forward)')
plt.loglog(h_values, h_squared * 1e1, 'r:', alpha=0.5, label='$\propto h^2$ (central)')

plt.xlabel('Step size h')
plt.ylabel('Absolute error in derivative')
plt.title(f"Numerical Derivative Error for f(x) = x(x-1) at x = {x_point}")
plt.legend()
plt.savefig('/Users/mattthew/Documents/GitHub/Comp-Sim-Numerical-Methods/Comp-Sim-Assignment-1/plots/numerical_derivatives.pdf')
plt.show()

# Question 1b: Can we achieve machine precision accuracy?
print("Question 1b: Achieving Machine Precision")

# Find optimal h for central difference
min_central_idx = np.argmin(central_errors)
optimal_h_central = h_values[min_central_idx]
min_central_error = central_errors[min_central_idx]

print(f"Central Difference Method:")
print(f"  Optimal h: {optimal_h_central:.5e}")
print(f"  Minimum error: {min_central_error:.5e}")
print(f"  Ratio to machine precision: {min_central_error / machine_precision:.2f}")

# Find optimal h for forward difference
min_forward_idx = np.argmin(forward_errors)
optimal_h_forward = h_values[min_forward_idx]
min_forward_error = forward_errors[min_forward_idx]

print(f"Forward Difference Method:")
print(f"  Optimal h: {optimal_h_forward:.5e}")
print(f"  Minimum error: {min_forward_error:.5e}")
print(f"  Ratio to machine precision: {min_forward_error / machine_precision:.2f}")
