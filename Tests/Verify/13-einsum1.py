import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
BackEnd._config.disable_gpu()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ij_j_ij = BackEnd._numpy.einsum_ij_j_ij
scipy_einsum_ij_j_ij = BackEnd._scipy.einsum_ij_j_ij
torch_einsum_ij_j_ij = BackEnd._torch.einsum_ij_j_ij

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
