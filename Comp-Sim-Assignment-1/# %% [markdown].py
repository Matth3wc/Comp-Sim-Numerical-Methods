# %% [markdown]
# A: Heron’s root finding method 
# Given a real number $a$, with $a > 0$, how does one find the value of $x$ so that $x^2
# = a$?
# One can use the following iterative process, possibly dating back to the Babylonians:
# 
# 
# Part 2: Plot $x_n$ as a function of the iteration number n (try two or three different choices
# of $x_0$). 
# 
# Part 3: Plot the relative error for the quantity $x_n \times x_n$ as a function of n.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def heron_root(a, x_0, n):
    x_n = x_0
    for _ in range(n):
        x_n = 0.5 * (x_n + a / x_n)
    return x_n

# %% [markdown]
# Part 1: $x_{n+1} = \frac{1}{2}(x_n +\frac{a}{x_n})$, where $x_0$ is a chosen starting value and n is the iteration number.
# Write Python code to compute the square root of $a = 2$.

# %%
sqrt_2 = heron_root(2, 1, 10)
print("Approximation of x=\sqrt{2} using Heron's method")

# %%
x_0_array = np.array([0.5,0.75,1, 1.25,1.5, 1.75, 2,2.25])
n_array = np.arange(1,len(x_0_array)+1)

# %%
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
        
    #plt.plot(n_array, heron_root_array, label='Heron\'s Method with $x_0$ = ' + str(x_0))
    #plt.plot(n_array, heron_root_array, marker='x')
    print(heron_root_array)
    print(n_array)

plt.xlabel(f'Iteration number (n)')
plt.ylabel('Approximation of $\sqrt{2}$')
plt.title("Convergence of Heron's Method to $x=\sqrt{2}$")
plt.axhline(y=np.sqrt(2), color='r', linestyle='--', label='np.sqrt(2) result')
plt.xlim(1,5)
plt.ylim(0,2.5)
plt.legend()
plt.show()

# %%
markers = ['x', '^', 'o', 's', 'D', 'v']  # extend if needed

plt.figure(figsize=(8, 6))
for i, x_0 in enumerate(x_0_array):
    y = np.array([heron_root(2, x_0, n-1) for n in n_array])

    line, = plt.plot(n_array, y, linewidth=1.8,
                     label=fr"$x_0$ = {x_0}")

    plt.plot(n_array, y, linestyle='None',
             marker=markers[i % len(markers)],
             markersize=8,
             markerfacecolor='none',
             markeredgewidth=1.6,
             color=line.get_color(),
             zorder=3)

plt.axhline(np.sqrt(2), color='r', linestyle='--', label='np.sqrt(2)')
plt.legend()
plt.show()

# %%
def heron_square_error(a, x_0, n):
    error = (heron_root(a, x_0, n)**2 - np.sqrt(a)**2)/np.sqrt(a)**2
    return error


# %% [markdown]
# Considering the scenario where $a=2$, $x_0=1$ and $n=10$
# 
# 

# %%
heron_square_error_array = np.array([heron_square_error(2, 2, n) for n in n_array])
print(heron_square_error_array)

# %%
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
        
    #plt.plot(n_array, heron_square_error_array, label='Heron\'s Method with $x_0$ = ' + str(x_0))
    #plt.plot(n_array, heron_square_error_array, marker='x')
    print(heron_square_error_array)
    print(n_array)

plt.xlabel(f'Iteration number (n)')
plt.ylabel('Approximation of $\sqrt{2}$')
plt.title("Relative error in Heron's Method to $x=\sqrt{2}$")
plt.xlim(1,5)
plt.legend()
plt.show()

# %%



