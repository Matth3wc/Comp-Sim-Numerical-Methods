'''import matplotlib.pyplot as plt

x_values = list(range(10))
b_values = [1, 2, 3]

def listfun1(x_values, b_values):
    results = []  # Initialize the 2D list to store results
    for b in b_values:   
        row = []  # Initialize a new row for each value of b
        for x in x_values:
            y = x ** b  # Corrected the formula to calculate x^b
            row.append(y)  # Append the calculated value to the row
        results.append(row)  # Append the row to the results
    return results

results = listfun1(x_values, b_values)

print('rows =', len(results))  # Number of rows corresponds to the number of b values
print('cols =', len(results[0]))  # Number of columns corresponds to the number of x values

# Transpose the results to match the plotting requirements
transposed_results = list(map(list, zip(*results)))

plt.plot(transposed_results)
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Plot of y = x^b for different b values')
plt.legend([f'b={b}' for b in b_values])
plt.show()'''

import matplotlib.pyplot as plt
import timeit
import psutil  # Install with `pip install psutil`
import tracemalloc  # For peak memory usage

x_values = list(range(10))
b_values = [1, 2, 3]

def listfun1(x_values, b_values):
    results = []  # Initialize the 2D list to store results
    for b in b_values:   
        row = []  # Initialize a new row for each value of b
        for x in x_values:
            y = x ** b  # Calculate x^b
            row.append(y)  # Append the calculated value to the row
        results.append(row)  # Append the row to the results
    return results

def list_metrics():


    # Start tracemalloc for peak memory usage
    #tracemalloc.start()

    # Measure execution time using timeit
    execution_time = timeit.timeit(lambda: listfun1(x_values, b_values), number=30) / 30

    # Measure CPU and memory usage during the loop
    process = psutil.Process()
    for _ in range(30):
        cpu_usage_list.append(process.cpu_percent(interval=0.1))  # Measure CPU usage during each iteration
        memory_usage_list.append(process.memory_info().rss)  # Measure memory usage during each iteration

    # Stop tracemalloc and get peak memory usage
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate average CPU and memory usage
    avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list)
    avg_memory_usage = sum(memory_usage_list) / len(memory_usage_list)

    return execution_time, avg_cpu_usage, peak_memory / 1024, avg_memory_usage / 1024

# Get performance metrics
execution_time, avg_cpu_usage, peak_memory, avg_memory_usage = list_metrics()

# Print computational burden
print(f"Average Execution Time: {execution_time:.6f} seconds")
print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")
print(f"Peak Memory Usage: {peak_memory:.2f} KB")
print(f"Average Memory Usage: {avg_memory_usage:.2f} KB")

# Transpose the results to match the plotting requirements
results = listfun1(x_values, b_values)
transposed_results = list(map(list, zip(*results)))

# Plot the results
plt.plot(transposed_results)
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Plot of y = x^b for different b values')
plt.legend([f'b={b}' for b in b_values])
plt.show()