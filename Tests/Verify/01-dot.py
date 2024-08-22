import numpy as np
import torch
import backend._config

backend._config.disable_fftw()
import backend._numpy
import backend._scipy
import backend._torch

backend._torch.disable_gpu()

numpy_dot = backend._numpy.dot
scipy_dot = backend._scipy.dot
torch_dot = backend._torch.dot


def test_dot_interface(dot_func, name):
    print(f"Testing {name} dot interface:")

    # Test case 1: Basic matrix multiplication
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    b = np.array([[5, 6], [7, 8]], dtype=np.float64)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float64)
    result = dot_func(a, b)
    assert np.allclose(result, expected), f"{name} failed basic matrix multiplication"
    print("  Passed basic matrix multiplication")

    # Test case 2: With alpha
    alpha = 2
    expected = np.array([[38, 44], [86, 100]], dtype=np.float64)
    result = dot_func(a, b, alpha=alpha)
    assert np.allclose(
        result, expected
    ), f"{name} failed matrix multiplication with alpha"
    print("  Passed matrix multiplication with alpha")

    # Test case 3: With c and beta
    c = np.array([[1, 1], [1, 1]], dtype=np.float64)
    beta = 3
    expected = np.array([[41, 47], [89, 103]], dtype=np.float64)
    result = dot_func(a, b, alpha=alpha, c=c, beta=beta)
    assert np.allclose(
        result, expected
    ), f"{name} failed matrix multiplication with c and beta"
    print("  Passed matrix multiplication with c and beta")

    # Test case 4: Default values (alpha=1, c=None, beta=0)
    result = dot_func(a, b, alpha=1, c=None, beta=0)
    assert np.allclose(
        result, np.array([[19, 22], [43, 50]])
    ), f"{name} failed with default values"
    print("  Passed matrix multiplication with default values")

    # Test case 5: Non-square matrices
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
    expected = np.array([[58, 64], [139, 154]], dtype=np.float64)
    result = dot_func(a, b)
    assert np.allclose(result, expected), f"{name} failed with non-square matrices"
    print("  Passed matrix multiplication with non-square matrices")

    print(f"{name} dot interface tests passed successfully!\n")


# Run tests for each dot implementation
test_dot_interface(numpy_dot, "NumPy")
test_dot_interface(scipy_dot, "SciPy")


# For PyTorch, we need to wrap the function to handle numpy arrays
def torch_dot_wrapper(a, b, alpha=1, c=None, beta=0):
    a_torch = torch.from_numpy(a)
    b_torch = torch.from_numpy(b)
    if c is not None:
        c_torch = torch.from_numpy(c)
    else:
        c_torch = None
    result = torch_dot(a_torch, b_torch, alpha=alpha, c=c_torch, beta=beta)
    return result.numpy()


test_dot_interface(torch_dot_wrapper, "PyTorch")
