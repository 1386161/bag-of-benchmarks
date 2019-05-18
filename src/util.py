"""
  Some utility functions useful for MPI parallel programming
"""
from mpi4py import MPI

#=============================================================================
# I/O Utilities


KIBI = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']
KILO = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

def num_bytes(b, kibi=True):
    div, t = (1024, KIBI) if kibi else (1000, KILO)
    i = 0
    while b > div - 1:
        b /= div
        i += 1
    return f"{b} {t[i]}"


def mprint(*args, end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == 0:
        print(*args, end=end, flush=True)