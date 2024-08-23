import numpy as np
import torch
import backend._config

backend._config.disable_fftw()
import backend._numpy
import backend._scipy
import backend._torch

backend._torch.disable_gpu()

numpy_einsum_ij_j_ij = backend._numpy.einsum_ij_j_ij
scipy_einsum_ij_j_ij = backend._scipy.einsum_ij_j_ij
torch_einsum_ij_j_ij = backend._torch.einsum_ij_j_ij

m, n = 10, 12

a = np.random.rand(m, n)
b = np.random.rand(n)
c = np.random.rand(m)

np_result = numpy_einsum_ij_j_ij(a, b)
scipy_result = scipy_einsum_ij_j_ij(a, b)

a_torch = torch.tensor(a)
b_torch = torch.tensor(b)
torch_result = torch_einsum_ij_j_ij(a_torch, b_torch)
torch_result = torch_result.numpy()

benchmark = np.einsum("ij,j->ij", a, b)

assert np.allclose(np_result, benchmark)
assert np.allclose(scipy_result, benchmark)
assert np.allclose(torch_result, benchmark)
