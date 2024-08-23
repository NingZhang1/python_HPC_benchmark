import torch, numpy, scipy
import backend._pyfftw as _pyfftw
from backend._config import ENABLE_FFTW, FORCE_PYSCF_LIB
from backend._malloc import __malloc

USE_GPU = True
FFT_CPU_USE_TORCH_ANYWAY = False
QR_PIVOTING_GPU_ANYWAY = False


def disable_gpu():
    global USE_GPU
    USE_GPU = False


def enable_gpu():
    global USE_GPU
    USE_GPU = True


def enable_fft_cpu():
    global FFT_CPU_USE_TORCH_ANYWAY
    FFT_CPU_USE_TORCH_ANYWAY = True


def disable_fft_cpu():
    global FFT_CPU_USE_TORCH_ANYWAY
    FFT_CPU_USE_TORCH_ANYWAY = False


if FORCE_PYSCF_LIB:
    try:
        from pyscf import lib

        PYSCF_LIB_FOUND = True
    except ImportError:
        PYSCF_LIB_FOUND = False

torch.set_grad_enabled(False)

# toTensor #


def toTensor(data, cpu=True):
    if USE_GPU and not cpu:
        return torch.tensor(data, device="cuda")
    else:
        return torch.tensor(data, device="cpu")


def toNumpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    assert isinstance(data, numpy.ndarray)
    return data


def malloc(shape, dtype, buf=None, offset=0, gpu=False):
    if buf is None:
        assert offset is None or offset == 0
        return torch.empty(shape, dtype=dtype, device="cuda" if gpu else "cpu")
    else:
        if gpu:
            assert buf.is_cuda
        else:
            assert not buf.is_cuda
        elmtsize = buf.element_size()
        assert offset % elmtsize == 0
        assert dtype == buf.dtype
        return buf.view(-1)[
            offset // elmtsize : offset // elmtsize + numpy.prod(shape)
        ].view(*shape)


# FFT on cpu/gpu #

if _pyfftw.FFTW_FOUND and ENABLE_FFTW and not FFT_CPU_USE_TORCH_ANYWAY:

    def _rfftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.rfftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)

    def _irfftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.irfftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)

    def _fftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.fftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)

    def _ifftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.ifftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.rfft(x, s=s, dim=axes)
        else:
            return _rfftn_cpu(
                x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
            )

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.irfft(x, s=s, dim=axes)
        else:
            return _irfftn_cpu(
                x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
            )

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.fft.fftn(x, s=s, dim=axes)
        else:
            return _fftn_cpu(
                x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
            )

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.fft.ifftn(x, s=s, dim=axes)
        else:
            return _ifftn_cpu(
                x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
            )

else:

    # print("Using torch for FFT")

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.rfft(x, s=s, dim=axes)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.irfft(x, s=s, dim=axes)

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.fft.fftn(x, s=s, dim=axes)

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.fft.ifftn(x, s=s, dim=axes)


# matmul #

if FORCE_PYSCF_LIB and PYSCF_LIB_FOUND:

    def dot(a, b, alpha=1, c=None, beta=0):
        # return lib.dot(a, b, alpha=alpha, c=c, beta=beta)
        if c is None:
            a_numpy = toNumpy(a)
            b_numpy = toNumpy(b)
            c_numpy = lib.dot(a_numpy, b_numpy, alpha=alpha, c=None, beta=beta)
            return toTensor(c_numpy)
        else:
            c_numpy = toNumpy(c)
            a_numpy = toNumpy(a)
            b_numpy = toNumpy(b)
            c_numpy = lib.dot(a_numpy, b_numpy, alpha=alpha, c=c_numpy, beta=beta)
            return toTensor(c_numpy)

else:

    def dot(a, b, alpha=1, c=None, beta=0):
        if c is not None:
            c *= beta
            c += alpha * torch.matmul(a, b)
            return c
        else:
            return alpha * torch.matmul(a, b)


# QR #

QR_MODE_MAPPING = {
    "full": "complete",
    "complete": "complete",
    "reduced": "reduced",
    "r": "r",
    "raw": "raw",
    "economic": "reduced",
}


def _qr_col_pivoting(A, tol=1e-8, max_rank=None, mode="r"):
    """
    Perform QR decomposition with column pivoting using PyTorch.

    Args:
    A (torch.Tensor): Input matrix of shape (m, n)
    tol (float): Tolerance for zero comparison

    Returns:
    Q (torch.Tensor): Orthogonal matrix of shape (m, m)
    R (torch.Tensor): Upper triangular matrix of shape (m, n)
    P (torch.Tensor): Permutation matrix of shape (n, n)
    """

    assert mode in QR_MODE_MAPPING.keys()

    mode = QR_MODE_MAPPING[mode]

    m, n = A.shape
    if max_rank is None:
        max_rank = min(m, n)

    with_Q = mode != "r"

    if with_Q:
        Q = torch.eye(m, device=A.device, dtype=A.dtype)
    else:
        Q = None
    R = A.clone()
    P = (
        torch.arange(start=0, end=n, device=A.device, dtype=torch.int32)
        .clone()
        .detach()
    )

    # buffer #
    buf_norm = torch.zeros((n,), device=A.device, dtype=A.dtype)
    buf_u = torch.zeros((m,), device=A.device, dtype=A.dtype)
    # end #

    for k in range(min(m, n, max_rank)):
        # Find the column with the largest norm
        norms = buf_norm[: n - k].view(-1)
        norms = torch.norm(R[k:, k:], dim=0, out=norms)
        j = (torch.argmax(norms) + k).item()

        # Swap columns
        R[:, [k, j]] = R[:, [j, k]]
        P[k], P[j] = P[j].item(), P[k].item()

        # Check if the remaining submatrix is negligible
        if norms[j - k] < tol:
            break

        # Compute Householder reflection
        x = R[k:, k]
        alpha = torch.norm(x).item()
        if x[0] < 0:
            alpha = -alpha
        u_view = buf_u[: m - k].view(-1)
        u_view.copy_(x)
        u = u_view
        u[0] += alpha
        u /= torch.norm(u).item()

        # Apply Householder reflection to R
        R[k:, k:] -= 2 * torch.outer(u, torch.matmul(u, R[k:, k:]))

        # Update Q
        if with_Q:
            Q[:, k:] -= 2 * torch.outer(torch.matmul(Q[:, k:], u), u)

    return Q, R, P


try:
    import backend._scipy

    def qr_col_pivoting(A, tol=1e-8, max_rank=None, mode="r"):
        if A.is_cuda:
            if QR_PIVOTING_GPU_ANYWAY:
                return _qr_col_pivoting(A, tol=tol, max_rank=max_rank, mode=mode)
            else:
                A_cpu = A.cpu()
                Q_cpu, R_cpu, P_cpu = qr_col_pivoting(
                    A_cpu, tol=tol, max_rank=max_rank, mode=mode
                )
                if Q_cpu is not None:
                    Q = Q_cpu.cuda()
                else:
                    Q = None
                R = R_cpu.cuda()
                P = P_cpu.cuda()
                return Q, R, P
        else:
            # return _qr_col_pivoting(A, tol=tol, max_rank=max_rank, mode=mode)
            a_numpy = toNumpy(A)
            if mode == "r":
                Q, R, P = backend._scipy.qr_col_pivoting(
                    a_numpy, tol=tol, max_rank=max_rank, mode=mode
                )
            else:
                Q, R, P = backend._scipy.qr_col_pivoting(
                    a_numpy, tol=tol, max_rank=max_rank, mode=mode
                )
                Q = toTensor(Q)
            R = toTensor(R)
            P = toTensor(P)
            return Q, R, P

except ImportError:

    def qr_col_pivoting(A, tol=1e-8, max_rank=None, mode="r"):
        return _qr_col_pivoting(A, tol=tol, max_rank=max_rank, mode=mode)


def qr(A, mode="complete"):
    mode = QR_MODE_MAPPING[mode]
    if mode == "r":
        return None, torch.linalg.qr(A, mode=mode)
    return torch.linalg.qr(A, mode=mode)


# index #


def index_add(A, dim, index, source, alpha=1):
    A.index_add_(dim, index, source, alpha=alpha)
    return A


def index_copy(A, dim, index, source):
    # equivalent to A[:,:,...,index,...:,...] = source
    A.index_copy_(dim, index, source)
    return A


def take(a, indices, axis=None, out=None):
    return torch.index_select(a, axis, indices, out=out)


# min/max/abs/norm #


def maximum(a, axis=None, out=None):
    return torch.max(a, dim=axis, out=out)


def minimum(a, axis=None, out=None):
    return torch.min(a, dim=axis, out=out)


def absolute(a, out=None):
    return torch.abs(a, out=out)


def Frobenius_norm(a):
    assert len(a.shape) == 2
    return torch.norm(a)


def einsum_ij_j_ij(a, b, out=None):
    if out is None:
        return a * b
    else:
        torch.mul(a, b, out=out)
        return out


def einsum_i_ij_ij(a, b, out=None):
    a_reshaped = a.unsqueeze(1)
    if out is None:
        return a_reshaped * b
    else:
        torch.mul(a_reshaped, b, out=out)
        return out


def einsum_ik_jk_ijk(A, B, out=None):

    # Get the shapes

    i, k1 = A.shape
    j, k2 = B.shape
    k = k1

    A_broadcast = A.unsqueeze(1)  # Shape: (i, 1, k)
    B_broadcast = B.unsqueeze(0)  # Shape: (1, j, k)

    if out is None:
        out = torch.empty((i, j, k), dtype=A.dtype, device=A.device)
    else:
        assert out.shape == (i, j, k), "Out tensor must have shape (i, j, k)"

    torch.mul(A_broadcast, B_broadcast, out=out)

    return out


# eigh #


def eigh(a):
    return torch.linalg.eigh(a)


# square #


def square(a, out=None):
    return torch.square(a, out=out)


def square_(a):
    return square(a, out=a)


# cholesky #


def cholesky(a, lower=True, overwrite_a=True, out=None):
    if overwrite_a:
        return torch.linalg.cholesky(a, upper=(not lower), out=a)
    else:
        return torch.linalg.cholesky(a, upper=(not lower), out=out)


# solve #


def solve_cholesky(
    a, b, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False
):
    c = cholesky(a, lower=lower, overwrite_a=overwrite_a)
    if overwrite_b:
        return torch.linalg.solve(c, b, upper=(not lower), out=b)
    else:
        return torch.linalg.solve(c, b, upper=(not lower))
