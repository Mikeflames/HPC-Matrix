# Matrix Multiplication with MPI

## Overview

This project demonstrates parallel matrix multiplication using MPI (Message Passing Interface) and NumPy. It includes both a sequential implementation and a parallel implementation to showcase the differences in performance. The project aims to explore the benefits and limitations of parallel computing for large-scale matrix operations.

## Project Structure

- `matrix_multiply.py`: Sequential matrix multiplication implementation using NumPy.
- `matrix_multiply_mpi.py`: Parallel matrix multiplication implementation using MPI and `mpi4py`.

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

4. ***Results***:
   - Sequential Matrix Multiplication
   - Execution Time: 0.863 seconds for a 5000x5000 matrix
   
   - Parallel Matrix Multiplication
   - Execution Time: 1.860 seconds for a 5000x5000 matrix with 4 processes

5. ***Observations***:
   The sequential implementation is faster than the parallel implementation for this matrix size and setup.
   Performance of the parallel implementation might be affected by communication overhead and resource contention.

6. ***Future Work***:
   - Test with larger matrices and different numbers of processes to better understand the performance characteristics of parallel matrix multiplication.
   - Optimize MPI communication and workload distribution to improve parallel performance.
   - Explore additional parallel computing techniques and libraries for enhanced performance.

7. ***Contact***
   For questions or further discussions, please contact:
   - Mishal Singhai [Email](mailto:mishalsinghai21032001@gmail.com)






