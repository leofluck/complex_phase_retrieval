from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
​
# Run with "python3 MPI_example.py". Should produce same output as parallel
def main_serial():
    size = 3
​
    x = np.linspace(0,1,1024)
    y_all = np.zeros([size, len(x)])
    for i in range(size):
        y_all[i] = np.sin(2*x*np.pi*i)
​
    np.savetxt("MPI_example_serial.csv", y_all.T, delimiter=",")
​
# Run with "mpiexec -n 3 python3 MPI_example.py". Should produce same output as serial
def main_parallel():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
​
    x = np.linspace(0,1,1024)
    y_local = np.sin(2*x*np.pi*rank)
​
    y_all = np.zeros([size, len(x)])
    comm.Gather(y_local, y_all, root=0)
    if rank == 0:
        np.savetxt("MPI_example_parallel.csv", y_all.T, delimiter=",")
    
​
​
​
​
if __name__=="__main__":
    main_serial()
    main_parallel()