#!/usr/bin/env python

from __future__ import division

from time import time
from mpi4py import MPI
import numpy as np

from util import mprint, num_bytes

sizes = [ 2**n for n in range(1,24) ]
runs = 50

comm = MPI.COMM_WORLD

mprint("Benchmarking Reduce performance on %d parallel MPI processes..." % comm.size)
mprint("%15s | %12s | %12s" % ("Size (bytes)", "Time (msec)", "Bandwidth"))

for s in sizes:
    data = np.ones(s)
    res = np.empty_like(data)

    comm.Barrier()
    t_min = np.inf
    for i in range(runs):
        t0 = time()
        comm.Reduce([data, MPI.DOUBLE], [res, MPI.DOUBLE])
        t = time()-t0
        t_min = min(t, t_min)
    comm.Barrier()

    mprint("%20s | %16.3f | %16s/s" % (num_bytes(data.nbytes), t_min * 1000, num_bytes(data.nbytes / t_min)))

mprint("Benchmarking AllReduce performance on %d parallel MPI processes..." % comm.size)
mprint("%20s | %16s | %16s" % ("Size (bytes)", "Time (msec)", "Bandwidth (MiBytes/s)"))

for s in sizes:
    data = np.ones(s)
    res = np.empty_like(data)

    comm.Barrier()
    t_min = np.inf
    for i in range(runs):
        t0 = time()
        comm.Allreduce([data, MPI.DOUBLE], [res, MPI.DOUBLE])
        t = time()-t0
        t_min = min(t, t_min)
    comm.Barrier()
    
    mprint("%20s | %16.3f | %16s/s" % (num_bytes(data.nbytes), t_min*1000, num_bytes(data.nbytes/t_min)) )

