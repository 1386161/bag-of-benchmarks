"""
  Some utility functions useful for MPI parallel programming
"""
from mpi4py import MPI
from termcolor import cprint

#=============================================================================
# I/O Utilities

def mprint(string="", end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    cprint(str(comm.rank) + ': ', end='')
    print(string + end, flush=True)
