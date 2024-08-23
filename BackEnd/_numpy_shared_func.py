# this file shared the functions shared by numpy and scipy backend
# to avoid mutual import

import numpy


def index_add(A, dim, index, source, alpha=1):
    A[(slice(None),) * dim + (index,)] += alpha * source
    return A


def index_copy(A, dim, index, source):
    # equivalent to A[:,:,...,index,...:,...] = source
    A[(slice(None),) * dim + (index,)] = source
    return A


def take(a, indices, axis=None, out=None):
    return numpy.take(a, indices, axis=axis, out=out)


def maximum(a, axis=None, out=None):
    return numpy.max(a, axis=axis, out=out)


def minimum(a, axis=None, out=None):
    return numpy.min(a, axis=axis, out=out)


def absolute(a, out=None):
    return numpy.abs(a, out=out)


def Frobenius_norm(a):
    assert a.ndim == 2
    return numpy.linalg.norm(a)


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


def square(a, out=None):
    return numpy.square(a, out=out)


def square_(a):
    return square(a, out=a)
