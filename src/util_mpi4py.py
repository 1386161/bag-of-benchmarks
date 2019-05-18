"""
  Some utility functions useful for MPI parallel programming
"""
from mpi4py import MPI

#=============================================================================
# I/O Utilities

def mprint(*args, end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == 0:
        print(*args, end=end, flush=True)