import numpy
import torch

import BackEnd._config

BackEnd._config.disable_fftw()

import BackEnd._numpy
import BackEnd._scipy
import BackEnd._pyfftw
import BackEnd._torch

BackEnd._torch.disable_gpu()

m, n, k = 13, 17, 19

a = numpy.random.rand(m, n, k)
a_torch = a.copy()
a_torch = BackEnd._torch.toTensor(a_torch)

b1 = BackEnd._numpy.fftn(a, s=(m, n, k), axes=(0, 1, 2))
b2 = BackEnd._scipy.fftn(a, s=(m, n, k), axes=(0, 1, 2))
b3 = BackEnd._pyfftw.fftn(a, s=(m, n, k), axes=(0, 1, 2))
b4 = BackEnd._torch.fftn(a_torch, s=(m, n, k), axes=(0, 1, 2))
assert isinstance(b1, numpy.ndarray)
assert isinstance(b2, numpy.ndarray)
assert isinstance(b3, numpy.ndarray)
assert isinstance(b4, torch.Tensor)
b4 = BackEnd._torch.toNumpy(b4)

assert numpy.allclose(b1, b2)
assert numpy.allclose(b2, b3)
assert numpy.allclose(b3, b4)
