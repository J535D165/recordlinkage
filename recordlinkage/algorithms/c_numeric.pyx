cimport cython

import numpy as np

cimport numpy as np


cdef extern from "../_lib/numeric.h":

    # numeric distance functions
    double euclidean_dist(double x, double y)
    double haversine_dist(double th1, double ph1, double th2, double ph2)

    # numeric similarity functions
    double step_sim(double d, double offset, double origin)
    double linear_sim(double d, double scale, double offset, double origin)
    double squared_sim(double d, double scale, double offset, double origin)
    double exp_sim(double d, double scale, double offset, double origin)
    double gauss_sim(double d, double scale, double offset, double origin)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def euclidean_distance(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):

    cdef int n_rows = x.shape[0]

    cdef np.ndarray result = np.zeros(n_rows, dtype=np.float64)

    for k in range(n_rows):
        result[k] = euclidean_dist(x[k], y[k])

    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def haversine_distance(np.ndarray[np.float64_t, ndim=1] th1, np.ndarray[np.float64_t, ndim=1] ph1, np.ndarray[np.float64_t, ndim=1] th2, np.ndarray[np.float64_t, ndim=1] ph2):

    cdef int n_rows = th1.shape[0]

    cdef np.ndarray result = np.zeros(n_rows, dtype=np.float64)

    for k in range(n_rows):
        result[k] = haversine_dist(th1[k], ph1[k], th2[k], ph2[k])

    return result
