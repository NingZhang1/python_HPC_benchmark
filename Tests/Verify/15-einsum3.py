import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
BackEnd._config.disable_gpu()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ik_jk_ijk = BackEnd._numpy.einsum_ik_jk_ijk
scipy_einsum_ik_jk_ijk = BackEnd._scipy.einsum_ik_jk_ijk
torch_einsum_ik_jk_ijk = BackEnd._torch.einsum_ik_jk_ijk

m, n, k = 11, 13, 17

a = np.random.rand(m, k)
b = np.random.rand(n, k)
c = np.random.rand(m, n, k)

np_result = numpy_einsum_ik_jk_ijk(a, b)
scipy_result = scipy_einsum_ik_jk_ijk(a, b)

a_torch = torch.tensor(a)
b_torch = torch.tensor(b)
torch_result = torch_einsum_ik_jk_ijk(a_torch, b_torch)
torch_result = torch_result.numpy()

benchmark = np.einsum("ik,jk->ijk", a, b)

assert np.allclose(np_result, benchmark)
assert np.allclose(scipy_result, benchmark)
assert np.allclose(torch_result, benchmark)
