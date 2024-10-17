from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

local_sum = rank + 1 ## rank 0 will have 1, rank 1 will have 2, and so on
global_sum = comm.reduce(local_sum, op = MPI.SUM, root = 0)

if rank == 0:
    print(f"Global sum: {global_sum}")