import numpy as np
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

if __name__ == "__main__":
    # Initialize matrices
    n = 5000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Start timing
    start_time = time.time()

    # Perform multiplication
    C = matrix_multiply(A, B)

    # End timing
    end_time = time.time()

    print("Matrix multiplication completed.")
    print("Execution Time:", end_time - start_time, "seconds")
