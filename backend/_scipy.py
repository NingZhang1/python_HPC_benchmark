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

# min/max/abs/norm #

from backend._numpy import maximum as numpy_max
from backend._numpy import minimum as numpy_min
from backend._numpy import absolute as numpy_abs
from backend._numpy import Frobenius_norm as numpy_Frobenius_norm

maximum = numpy_max
minimum = numpy_min
absolute = numpy_abs
Frobenius_norm = numpy_Frobenius_norm

# special einsum #

from backend._numpy import einsum_ij_j_ij as numpy_einsum_ij_j_ij
from backend._numpy import einsum_i_ij_ij as numpy_einsum_i_ij_ij
from backend._numpy import einsum_ik_jk_ijk as numpy_einsum_ik_jk_ijk
from backend._numpy import eigh as numpy_eigh

einsum_ij_j_ij = numpy_einsum_ij_j_ij
einsum_ik_jk_ijk = numpy_einsum_ik_jk_ijk

# eigh #


def eigh(a):
    return scipy.linalg.eigh(a)


# square #


from backend._numpy import square as numpy_square
from backedn._numpy import square_ as numpy_square_

square = numpy_square
square_ = numpy_square_

# cholesky #


def cholesky(a, lower=True, overwrite_a=True, out=None):
    return scipy.linalg.cholesky(
        a, lower=lower, overwrite_a=overwrite_a, check_finite=False
    )


# solve #


def solve_cholesky(
    a, b, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False
):
    C = cholesky(a, lower, overwrite_a, out=None)
    B = scipy.linalg.cho_solve(
        (C, lower), b, overwrite_b=overwrite_b, check_finite=check_finite
    )
    return B
