import numpy
import torch

import BackEnd._config

# BackEnd._config.disable_fftw()
BackEnd._config.disable_gpu()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._pyfftw
import BackEnd._torch


m, n, k = 64, 64, 64

a = numpy.random.rand(m, n, k)
a_torch = a.copy()
a_torch = BackEnd._torch.toTensor(a_torch)

b1 = BackEnd._numpy.fftn(a, s=(m, n, k), axes=(0, 1, 2), overwrite_input=True)
b2 = BackEnd._scipy.fftn(a, s=(m, n, k), axes=(0, 1, 2), overwrite_input=True)
b3 = BackEnd._pyfftw.fftn(a, s=(m, n, k), axes=(0, 1, 2), overwrite_input=True)
b4 = BackEnd._torch.fftn(a_torch, s=(m, n, k), axes=(0, 1, 2), overwrite_input=True)
assert isinstance(b1, numpy.ndarray)
assert isinstance(b2, numpy.ndarray)
assert isinstance(b3, numpy.ndarray)
assert isinstance(b4, torch.Tensor)

# print address #

print("data address of a: ", a.__array_interface__["data"][0])
print("data address of b1: ", b1.__array_interface__["data"][0])
print("data address of b2: ", b2.__array_interface__["data"][0])
print("data address of b3: ", b3.__array_interface__["data"][0])
print("data address of b4: ", b4.data_ptr())

b4 = BackEnd._torch.toNumpy(b4)

print("data address of b4: ", b4.__array_interface__["data"][0])

assert numpy.allclose(b1, b2)
assert numpy.allclose(b2, b3)
assert numpy.allclose(b3, b4)

import pyfftw

a2 = pyfftw.empty_aligned((m, n, k))
print("data address of a2 : ", a2.__array_interface__["data"][0])
a2 = numpy.asarray(a2)
print("data address of a2 : ", a2.__array_interface__["data"][0])
b22 = BackEnd._scipy.fftn(a2, s=(m, n, k), axes=(0, 1, 2), overwrite_input=True)
print("data address of b22: ", b22.__array_interface__["data"][0])
