import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._scipy
import BackEnd._torch

BackEnd._torch.disable_gpu()

numpy_einsum_i_ij_ij = BackEnd._numpy.einsum_i_ij_ij
scipy_einsum_i_ij_ij = BackEnd._scipy.einsum_i_ij_ij
torch_einsum_i_ij_ij = BackEnd._torch.einsum_i_ij_ij

m, n = 10, 12

a = np.random.rand(m)
b = np.random.rand(m, n)
c = np.random.rand(m, n)

np_result = numpy_einsum_i_ij_ij(a, b)
scipy_result = scipy_einsum_i_ij_ij(a, b)

a_torch = torch.tensor(a)
b_torch = torch.tensor(b)
torch_result = torch_einsum_i_ij_ij(a_torch, b_torch)
torch_result = torch_result.numpy()

benchmark = np.einsum("i,ij->ij", a, b)

assert np.allclose(np_result, benchmark)
assert np.allclose(scipy_result, benchmark)
assert np.allclose(torch_result, benchmark)
