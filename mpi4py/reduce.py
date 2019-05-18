#!/usr/bin/env python

from __future__ import division

from time import time
from mpi4py import MPI
import numpy as np

from pylib.parutils import mprint

sizes = [ 2**n for n in range(1,24) ]
runs = 50

comm = MPI.COMM_WORLD

mprint("Benchmarking Reduce performance on %d parallel MPI processes..." % comm.size)
mprint("%15s | %12s | %12s" % ("Size (bytes)", "Time (msec)", "Bandwidth (MiBytes/s)"))

for s in sizes:
    data = np.ones(s)
    res = np.empty_like(data)

    comm.Barrier()
    t_min = np.inf
    for i in range(runs):
        t0 = time()
        comm.Reduce( [data, MPI.DOUBLE], [res, MPI.DOUBLE] ) 
        t = time()-t0
        t_min = min(t, t_min)
    comm.Barrier()
    
    mprint("%15d | %12.3f | %12.3f" %
        (data.nbytes, t_min*1000, data.nbytes/t_min/1024/1024) )

mprint("Benchmarking AllReduce performance on %d parallel MPI processes..." % comm.size)
mprint("%15s | %12s | %12s" % ("Size (bytes)", "Time (msec)", "Bandwidth (MiBytes/s)"))

for s in sizes:
    data = np.ones(s)
    res = np.empty_like(data)

    comm.Barrier()
    t_min = np.inf
    for i in range(runs):
        t0 = time()
        comm.Allreduce( [data, MPI.DOUBLE], [res, MPI.DOUBLE] ) 
        t = time()-t0
        t_min = min(t, t_min)
    comm.Barrier()
    
    mprint("%15d | %12.3f | %12.3f" % (data.nbytes, t_min*1000, data.nbytes/t_min/1024/1024) )

