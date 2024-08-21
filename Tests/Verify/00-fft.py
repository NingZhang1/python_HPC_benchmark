import numpy

import backend._config

backend._config.disable_fftw()

import backend._numpy
import backend._scipy
import backend._pyfftw
import backend._torch

backend._torch.disable_gpu()

m, n, k = 13, 15, 17

a = numpy.random.rand(m, n, k)
a_torch = a.copy()
a_torch = backend._torch.toTensor(a_torch)

b1 = backend._numpy.fftn(a, s=(m, n, k), axes=(0, 1, 2))
b2 = backend._scipy.fftn(a, s=(m, n, k), axes=(0, 1, 2))
b3 = backend._pyfftw.fftn(a, s=(m, n, k), axes=(0, 1, 2))
b4 = backend._torch.fftn(a_torch, s=(m, n, k), axes=(0, 1, 2))
b4 = backend._torch.toNumpy(b4)

assert numpy.allclose(b1, b2)
assert numpy.allclose(b2, b3)
assert numpy.allclose(b3, b4)
