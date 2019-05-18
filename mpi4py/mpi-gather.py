#!/usr/bin/env python

from __future__ import division
import numpy as np
from mpi4py import MPI
from pylib.parutils import mprint

#=============================================================================
# Main

comm = MPI.COMM_WORLD

mprint("-"*78)
mprint(" Running %d parallel processes..." % comm.size)
mprint("-"*78)

my_N = 10 + comm.rank
my_a = comm.rank * np.ones(my_N)
N = comm.allreduce(my_N)
a = comm.gather(my_a)

mprint("Gathered array: %s" % a)
