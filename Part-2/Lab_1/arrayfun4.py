import numpy as np
import time
import sys

def arrayfun4(x_values, b_values):
    """
    Optimized function to calculate y = x^b for three integer values of b
    and a range of 1,000 integer x-values using NumPy arrays.
    """
    # Convert x_values and b_values to NumPy arrays
    x_values = np.array(x_values).reshape(-1, 1)  # Reshape x_values to a column vector
    b_values = np.array(b_values)  # Convert b_values to a NumPy array

    # Use broadcasting to compute y = x^b for all combinations of x and b
    results = x_values ** b_values  # Vectorized computation
    return results

if __name__ == "__main__":
    # Define x_values and b_values
    x_values = np.arange(1000)  # x values from 0 to 999
    b_values = np.array([1, 2, 3])  # b values

    # Measure execution time
    execution_times = []
    memory_usages = []

    for _ in range(5):  # Run the performance check 5 times
        start_time = time.time()  # Start timing
        array_results = arrayfun4(x_values, b_values)
        end_time = time.time()  # End timing

        execution_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        execution_times.append(execution_time_ms)

        # Measure memory usage
        array_pointer_size = sys.getsizeof(array_results)  # Shallow memory of the array object
        array_memory_size = array_results.nbytes  # Memory used by the data in the array
        total_memory_usage = array_pointer_size + array_memory_size
        memory_usages.append(total_memory_usage)

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