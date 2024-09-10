import numpy as np
from mpi4py import MPI

def parallel_matrix_multiply(A, B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = A.shape[0]
    C = np.zeros((n, n))
    
    rows_per_process = n // size
    start_row = rank * rows_per_process
    end_row = (rank + 1) * rows_per_process if rank != size - 1 else n

    local_A = np.zeros((rows_per_process, n))
    comm.Scatter(A, local_A, root=0)
    comm.Bcast(B, root=0)

    local_C = np.dot(local_A, B)
    comm.Gather(local_C, C, root=0)

    return C

if __name__ == "__main__":
    n = 5000  # Adjust matrix size as needed
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start_time = MPI.Wtime()
    C = parallel_matrix_multiply(A, B)
    end_time = MPI.Wtime()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Execution Time:", end_time - start_time)
