# Matrix Multiplication with MPI and Matrix Decomposition

## Overview

This project demonstrates parallel matrix multiplication using MPI (Message Passing Interface) and NumPy. It includes both a sequential implementation and a parallel implementation to showcase the differences in performance. The project aims to explore the benefits and limitations of parallel computing for large-scale matrix operations.

For Matrix Decomposition :- LU Decomposition or Cholesky Decomposition can be parallelized. These techniques are used to break down a matrix into simpler, lower-dimensional matrices and are essential in solving systems of linear equations, a common operation in scientific computing and HPC.


Problem: Parallel LU Decomposition
LU decomposition factors a matrix 𝐴 into the product of a lower triangular matrix L and an upper triangular matrix U. 
This process can be computationally expensive for large matrices and can benefit from parallelization.

## Project Structure

- `matrix_multiply.py`: Sequential matrix multiplication implementation using NumPy.
- `matrix_multiply_mpi.py`: Parallel matrix multiplication implementation using MPI and `mpi4py`.
- `parallel_lu_decomposition.py1`: basic implementation of parallel LU decomposition using MPI

## Requirements

- Python 3.x
- NumPy
- mpi4py
- MPI implementation (e.g., OpenMPI or MPICH)

## Setup

1. **Install Dependencies**

   Install the required Python packages using pip:

   ```bash
   pip install numpy mpi4py

2. **Install MPI**

    Ensure that MPI is installed on your system. You can install OpenMPI or MPICH using your package manager.

    For Ubuntu/Debian:
    ```bash
    sudo apt-get install openmpi-bin libopenmpi-dev
    ```

    For macOS:
     ```bash
    brew install open-mpi
     ```

3. **Running the Code**:

    ***Sequential Implementation***
    Run the sequential matrix multiplication script:
    ```bash
    python3 matrix_multiply.py
    ```

    ***Parallel Implementation***
    Run the parallel matrix multiplication script using MPI:
    ```bash
    mpiexec -n 4 python3 matrix_multiply_mpi.py
    ```

    ***Parallel LU Decomposition with MPI***
   ```bash
   mpiexec -n 4 python parallel_lu_decomposition.py
    ```
4. **BreakDown for Parallel LU Decomposition**:
   - Process Distribution: The matrix rows are distributed across the processes, and each process handles its subset of rows.
   - Pivoting: The pivot row is broadcast to all processes, and each process uses this information to perform elimination.
   - Synchronization: Each step involves synchronizing the processes using MPI.COMM_WORLD.Bcast.
  
5. **Improvements and Challenges for Matrix Decomposition:**:
   - Load Balancing: Ensure the matrix rows are evenly distributed across processes.
   - Communication Overhead: Reducing the amount of communication is essential for better performance.
   - Scalability: Test with larger matrices (e.g., 10,000 x 10,000) to explore how the parallel decomposition scales as the number of processes increases.

6. ***Results***:
   - Sequential Matrix Multiplication
   - Execution Time: 0.863 seconds for a 5000x5000 matrix
   
   - Parallel Matrix Multiplication
   - Execution Time: 1.860 seconds for a 5000x5000 matrix with 4 processes

7. ***Observations***:
   The sequential implementation is faster than the parallel implementation for this matrix size and setup.
   Performance of the parallel implementation might be affected by communication overhead and resource contention.

8. ***Future Work***:
   - Test with larger matrices and different numbers of processes to better understand the performance characteristics of parallel matrix multiplication.
   - Optimize MPI communication and workload distribution to improve parallel performance.
   - Explore additional parallel computing techniques and libraries for enhanced performance.

9. ***Learnings***:
      This problem can serve as a good exploration of the trade-offs between computation and communication in parallel computing. We can extend this with optimizations like overlapping communication and computation, optimizing data locality, or even comparing the 
      performance of LU decomposition with different matrix sizes and process counts.

10. ***Contact***
   For questions or further discussions, please contact:
   - Mishal Singhai 
   [Email](mailto:mishalsinghai21032001@gmail.com)






