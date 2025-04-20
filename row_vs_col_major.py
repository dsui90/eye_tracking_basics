import numpy as np
from time import time

number_of_elements = 5000

A = np.zeros((number_of_elements, number_of_elements))
B = np.zeros((number_of_elements, number_of_elements))

# Row-major order
start_time = time()
for i in range(number_of_elements):
    for j in range(number_of_elements):
        A[i, j] = i + j
end_time = time()
print(f"Row-major order loop time: {end_time - start_time:.2f} seconds")

# Column-major order
start_time = time()
for j in range(number_of_elements):
    for i in range(number_of_elements):
        B[i, j] = i + j
end_time = time()
print(f"Column-major order loop time: {end_time - start_time:.2f} seconds")