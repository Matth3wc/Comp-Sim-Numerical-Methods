import time  # For timing
import sys   # For shallow memory usage

# Function to calculate y = x^b
def listfun2b(x_values, b_values):
    results = []  # Initialize the 2D list to store results
    for b in b_values:   
        row = []  # Initialize a new row for each value of b
        for x in x_values:
            y = x ** b  # Calculate x^b
            row.append(y)  # Append the calculated value to the row
        results.append(row)  # Append the row to the results
    return results

# Recursive function to calculate total memory usage
def calculate_total_memory(obj):
    """Recursively calculate the total memory usage of an object."""
    seen_ids = set()  # Track object IDs to avoid double-counting

    def inner(obj):
        obj_id = id(obj)
        if obj_id in seen_ids:  # Avoid double-counting shared references
            return 0
        seen_ids.add(obj_id)

        size = sys.getsizeof(obj)  # Shallow memory of the object
        if isinstance(obj, dict):
            size += sum(inner(k) + inner(v) for k, v in obj.items())  # Memory of keys and values
        elif isinstance(obj, (list, tuple, set)):
            size += sum(inner(i) for i in obj)  # Memory of elements
        return size

    return inner(obj)

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
        results = listfun2b(x_values, b_values)
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