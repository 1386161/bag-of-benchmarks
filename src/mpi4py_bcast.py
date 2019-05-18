#!/usr/bin/env python

from __future__ import division

from time import time
from mpi4py import MPI
import numpy as np

from .util_mpi4py import mprint

sizes = [ 2**n for n in range(1,24) ]
runs  = 20
comm = MPI.COMM_WORLD

mprint("Benchmarking braodcast performance on %d parallel MPI processes..." % comm.size)
mprint("%15s | %12s | %12s" % ("Size (bytes)", "Time (msec)", "Bandwidth (MiBytes/s)"))

for s in sizes:
    data = np.ones(s)

    comm.Barrier()
    t0 = time()
    for i in range(runs):
        comm.Bcast( [data, MPI.DOUBLE], 0)
    comm.Barrier()
    t = (time()-t0) / runs
    
    mprint("%15d | %12.3f | %12.3f" % (data.nbytes, t*1000, data.nbytes/t/1024/1024))
