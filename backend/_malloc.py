import numpy as np


try:
    import pyfftw

    def isdf_malloc(shape, dtype, buf=None, offset=0):
        if buf is None:
            return pyfftw.empty_aligned(shape, dtype=dtype)
        else:
            return np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)

except ImportError:

    def isdf_malloc(shape, dtype, buf=None, offset=0):
        if buf is None:
            return np.ndarray(shape, dtype=dtype)
        else:
            return np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)
