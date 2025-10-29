import numpy as np
import time
import sys

x_values = list(range(1_000))
b_values = [1,2,3]

def arrayfun3(x_values,b_values):
    # initialize 2D array results, with fixed number of columns but no rows yet
    array_results = np.empty((0,len(b_values)))
    for x in x_values:
        # initialize empty 1D array, row
        row = np.empty((0,))
        for b in b_values:
            y = x**b
            row = np.append(row,y)
        array_results = np.append(array_results,[row],axis=0)
    return array_results 

start_time = time.time()
array_results = arrayfun3(x_values,b_values)
end_time = time.time()

execution_time = (end_time-start_time)
print('Execution time = ', round(execution_time*1000),' ms')

array_pointer_size = sys.getsizeof(array_results)
array_memory_size = 1_000*array_results.itemsize

print('Pointer usage =',round(array_pointer_size/1024),'kB')
print('Memory usage =',round(array_memory_size/(1024)),'kB')
print('Total memory usage = ',round((array_pointer_size + array_memory_size)/(1024)),'kB')


if __name__ == "__main__":
    # Increase the range of x_values
    x_values = list(range(1000))  # x values from 0 to 999
    b_values = [1, 2, 3]

    # Lists to store execution times and memory usages for multiple runs
    execution_times = []
    memory_usages = []

    # Run the performance check 5 times
    for _ in range(5):
        # Measure execution time
        start_time = time.time()  # Start timing
        results = arrayfun3(x_values, b_values)
        end_time = time.time()  # End timing

        execution_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        execution_times.append(execution_time_ms)

        # Measure total memory usage
        total_memory_bytes = calculate_total_memory(results)
        memory_usages.append(total_memory_bytes)

    # Calculate averages
    avg_execution_time = sum(execution_times) / len(execution_times)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    # Print the last set of performance parameters
    print("\nPerformance Metrics (Last Run):")
    print("-" * 30)
    print(f"Last Execution Time : {execution_times[-1]:>10.2f} ms")
    print(f"Last Memory Usage   : {memory_usages[-1] / 1024:>10.2f} kB")
    print("-" * 30)

    # Print average performance metrics
    print("\nAverage Metrics (Over 5 Runs):")
    print("-" * 30)
    print(f"Avg Execution Time  : {avg_execution_time:>10.2f} ms")
    print(f"Avg Memory Usage    : {avg_memory_usage / 1024:>10.2f} kB")
    print("-" * 30)