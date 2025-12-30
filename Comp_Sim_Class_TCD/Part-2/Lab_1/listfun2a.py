import time  # For timing
import sys   # For memory usage

# Updated function
def listfun2a(x_values, b_values):
    results = []  # Initialize the 2D list to store results
    for b in b_values:   
        row = []  # Initialize a new row for each value of b
        for x in x_values:
            y = x ** b  # Calculate x^b
            row.append(y)  # Append the calculated value to the row
        results.append(row)  # Append the row to the results
    return results

if __name__ == "__main__":
    x_values = list(range(1000))
    b_values = [1, 2, 3]

    execution_times = []
    memory_usages = []

    for _ in range(5):  # Run the code 5 times
        start_time = time.time()
        results = listfun2a(x_values, b_values)
        end_time = time.time()

        execution_time_ms = (end_time - start_time) * 1000
        memory_usage_bytes = sys.getsizeof(results)

        execution_times.append(execution_time_ms)
        memory_usages.append(memory_usage_bytes)



    # Print the last set of performance parameters
    print("\nPerformance Metrics:")
    print("-" * 30)
    print(f"Last Execution Time : {execution_times[-1]:>10.2f} ms")
    print(f"Last Memory Usage   : {memory_usages[-1] / 1024:>10.2f} kB")
    print("-" * 30)

    # Optionally, print average performance metrics
    print("Average Metrics:")
    print("-" * 30)
    print(f"Avg Execution Time  : {sum(execution_times) / len(execution_times):>10.2f} ms")
    print(f"Avg Memory Usage    : {sum(memory_usages) / len(memory_usages) / 1024:>10.2f} kB")
    print("-" * 30)