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

