import matplotlib.pyplot as plt

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
plt.show()