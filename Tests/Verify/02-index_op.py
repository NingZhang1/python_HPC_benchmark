import numpy as np
import torch
import backend._config

backend._config.disable_fftw()
import backend._numpy
import backend._scipy
import backend._torch

backend._torch.disable_gpu()

numpy_index_add = backend._numpy.index_add
scipy_index_add = backend._scipy.index_add
torch_index_add = backend._torch.index_add

numpy_index_copy = backend._numpy.index_copy
scipy_index_copy = backend._scipy.index_copy
torch_index_copy = backend._torch.index_copy

numpy_take = backend._numpy.take
scipy_take = backend._scipy.take
torch_take = backend._torch.take


def verify_index_add(A_np, dim, index, source, alpha=1):
    A_scipy = A_np.copy()
    A_torch = torch.from_numpy(A_np.copy())
    source_torch = torch.from_numpy(source)
    index_torch = torch.from_numpy(index)

    numpy_index_add(A_np, dim, index, source, alpha=alpha)
    scipy_index_add(A_scipy, dim, index, source, alpha=alpha)
    torch_index_add(A_torch, dim, index_torch, source_torch, alpha=alpha)

    assert np.allclose(A_np, A_scipy), "NumPy and SciPy index_add results do not match"
    assert np.allclose(
        A_np, A_torch.numpy()
    ), "NumPy and PyTorch index_add results do not match"
    print("index_add verification passed")


def verify_index_copy(A_np, dim, index, source):
    A_scipy = A_np.copy()
    A_torch = torch.from_numpy(A_np.copy())
    source_torch = torch.from_numpy(source)
    index_torch = torch.from_numpy(index)

    numpy_index_copy(A_np, dim, index, source)
    scipy_index_copy(A_scipy, dim, index, source)
    torch_index_copy(A_torch, dim, index_torch, source_torch)

    assert np.allclose(A_np, A_scipy), "NumPy and SciPy index_copy results do not match"
    assert np.allclose(
        A_np, A_torch.numpy()
    ), "NumPy and PyTorch index_copy results do not match"
    print("index_copy verification passed")


def verify_take(a_np, indices, axis=None):
    result_numpy = numpy_take(a_np, indices, axis=axis)
    result_scipy = scipy_take(a_np, indices, axis=axis)

    a_torch = torch.from_numpy(a_np)
    indices_torch = torch.from_numpy(indices)
    result_torch = torch_take(a_torch, indices_torch, axis=axis).numpy()

    assert np.allclose(
        result_numpy, result_scipy
    ), "NumPy and SciPy take results do not match"
    assert np.allclose(
        result_numpy, result_torch
    ), "NumPy and PyTorch take results do not match"
    print("take verification passed")


def run_verifications():
    # Test case 1: 2D array, add to rows
    A = np.random.rand(5, 3)
    index = np.array([1, 3])
    source = np.random.rand(2, 3)
    verify_index_add(A, 0, index, source)
    verify_index_copy(A, 0, index, source)

    # Test case 2: 2D array, add to columns
    A = np.random.rand(3, 5)
    index = np.array([0, 2, 4])
    source = np.random.rand(3, 3)
    verify_index_add(A, 1, index, source)
    verify_index_copy(A, 1, index, source)

    # Test case 3: 3D array
    A = np.random.rand(4, 3, 5)
    index = np.array([1, 2])
    source = np.random.rand(2, 3, 5)
    verify_index_add(A, 0, index, source)
    verify_index_copy(A, 0, index, source)

    # Test case 4: with alpha
    A = np.random.rand(5, 3)
    index = np.array([1, 3])
    source = np.random.rand(2, 3)
    alpha = 0.5
    verify_index_add(A, 0, index, source, alpha)

    # Test case 5: take function
    a = np.random.rand(10, 5)
    indices = np.array([1, 3, 4])
    verify_take(a, indices, axis=0)
    verify_take(a, indices, axis=1)

    print("All verifications passed successfully!")


if __name__ == "__main__":
    run_verifications()
