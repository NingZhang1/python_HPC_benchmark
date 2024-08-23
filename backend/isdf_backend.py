from backend._config import USE_NUMPY, USE_SCIPY, USE_TORCH, USE_TORCH_GPU

# import different backend #

if USE_NUMPY:
    import backend._numpy as backend
elif USE_SCIPY:
    import backend._scipy as backend
elif USE_TORCH_GPU:
    import backend._torch as backend

    backend.enable_gpu()
else:
    import backend._torch as backend

    backend.disable_gpu()

# assign python interface #

_toTensor = backend.toTensor
_toNumpy = backend.toNumpy
_fft = backend.fft
_ifft = backend.ifft
_fftn = backend.fftn
_ifftn = backend.ifftn
_dot = backend.dot
_qr_col_pivoting = backend.qr_col_pivoting
_qr = backend.qr
_index_add = backend.index_add
_index_copy = backend.index_copy
_take = backend.take
_maximum = backend.maximum
_minimum = backend.minimum
_absolute = backend.absolute
_Frobenius_norm = backend.Frobenius_norm
_einsum_ij_j_ij = backend.einsum_ij_j_ij
_einsum_i_ij_ij = backend.einsum_i_ij_ij
_einsum_ik_jk_ijk = backend.einsum_ik_jk_ijk
_eigh = backend.eigh
_square = backend.square
_square_ = backend.square_
_cholesky = backend.cholesky
_solve_cholesky = backend.solve_cholesky
