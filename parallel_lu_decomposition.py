from mpi4py import MPI
import numpy as np

def lu_decomposition_parallel(A, rank, size):
    """
    Perform LU decomposition in parallel using MPI.
    A is the matrix to be decomposed. Each process gets a subset of rows to work on.
    rank: rank of the process
    size: total number of processes
    """
    n = A.shape[0]
    
    for k in range(n):
        if rank == k % size:
            pivot_row = A[k, :].copy()  # The pivot row
        else:
            pivot_row = np.empty(n)

        # Broadcast the pivot row to all processes
        MPI.COMM_WORLD.Bcast(pivot_row, root=k % size)

        # Each process updates its rows
        for i in range(k + 1, n):
            if i % size == rank:
                A[i, k] /= pivot_row[k]  # Compute the multiplier
                A[i, k + 1:] -= A[i, k] * pivot_row[k + 1:]  # Update the row

    return A

def generate_matrix(n):
    """Generate a random n x n matrix"""
    return np.random.rand(n, n)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Matrix size
    n = 8

    if rank == 0:
        A = generate_matrix(n)
        print("Original Matrix A:\n", A)
    else:
        A = None

    # Broadcast the matrix A to all processes
    A = comm.bcast(A, root=0)

    # Perform LU Decomposition in parallel
    A_decomposed = lu_decomposition_parallel(A, rank, size)

    if rank == 0:
        print("LU Decomposed Matrix:\n", A_decomposed)
