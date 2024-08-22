import numpy
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
    # print("Using pyfftw")

    rfftn = _pyfftw.rfftn
    irfftn = _pyfftw.irfftn
    fftn = _pyfftw.fftn
    ifftn = _pyfftw.ifftn

else:
    # print("Using numpy.fft")

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.rfftn(x, s, axes)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.irfftn(x, s, axes)

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.fftn(x, s, axes)

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.ifftn(x, s, axes)


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

try:
    import backend._scipy

    qr_col_pivoting = backend._scipy.qr_col_pivoting

except ImportError:

    try:
        import backend._torch

        def qr_col_pivoting(A, tol=1e-8, max_rank=None, mode="r"):
            A = backend._torch.toTensor(A)
            Q, R, P = backend._torch.qr_col_pivoting(
                A, tol=tol, max_rank=max_rank, mode=mode
            )
            if Q is not None:
                Q = backend._torch.toNumpy(Q)
            R = backend._torch.toNumpy(R)
            P = backend._torch.toNumpy(P)
            return Q, R, P

    except ImportError:
        raise ImportError("No backend available for qr with col pivoting")

QR_MODE_MAPPING = {
    "full": "complete",
    "complete": "complete",
    "reduced": "reduced",
    "r": "r",
    "raw": "raw",
    "economic": "reduced",
}


def qr(A, mode="complete"):
    return numpy.linalg.qr(A, mode=QR_MODE_MAPPING[mode])


# index #


def index_add(A, dim, index, source, alpha=1):
    A[(slice(None),) * dim + (index,)] += alpha * source
    return A


def index_copy(A, dim, index, source):
    # equivalent to A[:,:,...,index,...:,...] = source
    A[(slice(None),) * dim + (index,)] = source
    return A


def take(a, indices, axis=None, out=None):
    return numpy.take(a, indices, axis=axis, out=out)
