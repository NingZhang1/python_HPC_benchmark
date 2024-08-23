import numpy
import BackEnd._pyfftw as _pyfftw
from BackEnd._config import ENABLE_FFTW, ENABLE_PYSCF_LIB
from BackEnd._malloc import __malloc

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


malloc = __malloc

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
    import BackEnd._scipy as _scipy

    qr_col_pivoting = _scipy.qr_col_pivoting

except ImportError:

    try:
        import BackEnd._torch

        def qr_col_pivoting(A, tol=1e-8, max_rank=None, mode="r"):
            A = BackEnd._torch.toTensor(A)
            Q, R, P = BackEnd._torch.qr_col_pivoting(
                A, tol=tol, max_rank=max_rank, mode=mode
            )
            if Q is not None:
                Q = BackEnd._torch.toNumpy(Q)
            R = BackEnd._torch.toNumpy(R)
            P = BackEnd._torch.toNumpy(P)
            return Q, R, P

    except ImportError:
        raise ImportError("No BackEnd available for qr with col pivoting")

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


# min/max/abs/norm #


def maximum(a, axis=None, out=None):
    return numpy.max(a, axis=axis, out=out)


def minimum(a, axis=None, out=None):
    return numpy.min(a, axis=axis, out=out)


def absolute(a, out=None):
    return numpy.abs(a, out=out)


def Frobenius_norm(a):
    assert a.ndim == 2
    return numpy.linalg.norm(a)


# special einsum #


def einsum_ij_j_ij(a, b, out=None):
    if out is None:
        return a * b
    else:
        numpy.multiply(a, b, out=out)
        return out


def einsum_i_ij_ij(a, b, out=None):
    a_reshaped = a[:, numpy.newaxis]
    if out is None:
        return a_reshaped * b
    else:
        numpy.multiply(a_reshaped, b, out=out)
        return out


def einsum_ik_jk_ijk(a, b, out=None):
    if out is None:
        return numpy.einsum("ik,jk->ijk", a, b)
    else:
        return numpy.einsum("ik,jk->ijk", a, b, out=out)


# eigh #


def eigh(a):
    return numpy.linalg.eigh(a)


# square #


def square(a, out=None):
    return numpy.square(a, out=out)


def square_(a):
    return square(a, out=a)


# cholesky #


def cholesky(a, lower=True, overwrite_a=True, out=None):
    return numpy.linalg.cholesky(a)


# solve #

try:
    from BackEnd._scipy import solve_cholesky as scipy_solve_cholesky

    solve_cholesky = scipy_solve_cholesky

except ImportError:

    def solve_cholesky(
        a, b, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False
    ):
        return numpy.linalg.solve(a, b)
