import time
import sys

def listfun1(x_values, b_values):
    results = []
    for b in b_values:   
        row = []
        for x in x_values:
            y = x ** b
            row.append(y)
        results.append(row)
    return results

if __name__ == "__main__":
    x_values = list(range(10))
    b_values = [1, 2, 3]

    execution_times = []
    memory_usages = []

    for _ in range(5):  # Run the code 5 times
        start_time = time.time()
        results = listfun1(x_values, b_values)
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