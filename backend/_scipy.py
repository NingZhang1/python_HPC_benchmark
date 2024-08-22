import numpy, scipy
import backend._pyfftw as _pyfftw
from backend._config import ENABLE_FFTW, ENABLE_PYSCF_LIB

if ENABLE_PYSCF_LIB:
    try:
        from pyscf import lib

        PYSCF_LIB_FOUND = True
    except ImportError:
        PYSCF_LIB_FOUND = False

# toTensor #


def toTensor(data, cpu=True):
    assert cpu
    return numpy.asarray(data)


def toNumpy(data):
    if not isinstance(data, numpy.ndarray):
        raise ValueError("Data is not a numpy array")
    return data


# FFT #


if _pyfftw.FFTW_FOUND and ENABLE_FFTW:

    rfftn = _pyfftw.rfftn
    irfftn = _pyfftw.irfftn
    fftn = _pyfftw.fftn
    ifftn = _pyfftw.ifftn

else:

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.rfftn(x, s, axes, overwrite_x=overwrite_input, workers=threads)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.irfftn(
            x, s, axes, overwrite_x=overwrite_input, workers=threads
        )

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.fftn(x, s, axes, overwrite_x=overwrite_input, workers=threads)

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.ifftn(x, s, axes)


# matmul #

if ENABLE_PYSCF_LIB and PYSCF_LIB_FOUND:

    def dot(a, b, alpha=1, c=None, beta=0):
        return lib.dot(a, b, alpha=alpha, c=c, beta=beta)

else:

    def dot(a, b, alpha=1, c=None, beta=0):
        if c is not None:
            c *= beta
            c += alpha * numpy.dot(a, b)
            return c
        else:
            return alpha * numpy.dot(a, b)


# QR #

QR_MODE_MAPPING = {
    "full": "full",
    "complete": "full",
    "reduced": "economic",
    "r": "r",
    "raw": "raw",
    "economic": "economic",
}


def qr_col_pivoting(A, tol=1e-8, max_rank=None, mode="r"):
    if mode != "r":
        return scipy.linalg.qr(A, mode=QR_MODE_MAPPING[mode], pivoting=True)
    else:
        r, P = scipy.linalg.qr(A, mode=QR_MODE_MAPPING[mode], pivoting=True)
        return None, r, P


def qr(A, mode="full"):
    return scipy.linalg.qr(A, mode=QR_MODE_MAPPING[mode], pivoting=False)


# index #

from backend._numpy import index_add as numpy_index_add
from backend._numpy import index_copy as numpy_index_copy
from backend._numpy import take as numpy_take

index_add = numpy_index_add
index_copy = numpy_index_copy
take = numpy_take
