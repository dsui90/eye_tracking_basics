import numpy as np
from time import time

num_rows = 2
num_cols = 5000000

A = np.zeros((num_rows, num_cols))
B = np.zeros((num_rows, num_cols))

# Row-major order
start_time = time()
for i in range(num_rows):
    for j in range(num_cols):
        A[i, j] = i + j
end_time = time()
print(f"Row-major order loop time: {end_time - start_time:.2f} seconds")

# Column-major order
start_time = time()
for j in range(num_cols):
    for i in range(num_rows):
        B[i, j] = i + j
end_time = time()
print(f"Column-major order loop time: {end_time - start_time:.2f} seconds")