from mpi4py import MPI 


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'key1' : [7, 2.72, 2+3j],
            'key2' : ( 'abc', 'xyz')}
    comm.send(data, dest=1, tag = 11)  ## send data to rank 1, tag 11
elif rank == 1:
    data = comm.recv(source = 0, tag = 11)
    print(f"Rank 1 received: {data}")