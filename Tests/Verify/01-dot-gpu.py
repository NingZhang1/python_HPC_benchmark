import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._torch

torch_dot = BackEnd._torch.dot


def test_dot_interface(name, device="cpu"):
    print(f"Testing {name} dot interface on {device}:")

    # Test case 1: Basic matrix multiplication
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    b = np.array([[5, 6], [7, 8]], dtype=np.float64)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float64)
    result = torch_wrapper(a, b, device=device)
    assert np.allclose(
        result, expected
    ), f"{name} failed basic matrix multiplication on {device}"
    print("  Passed basic matrix multiplication")

    # Test case 2: With alpha
    alpha = 2
    expected = np.array([[38, 44], [86, 100]], dtype=np.float64)
    result = torch_wrapper(a, b, alpha=alpha, device=device)
    assert np.allclose(
        result, expected
    ), f"{name} failed matrix multiplication with alpha on {device}"
    print("  Passed matrix multiplication with alpha")

    # Test case 3: With c and beta
    c = np.array([[1, 1], [1, 1]], dtype=np.float64)
    beta = 3
    expected = np.array([[41, 47], [89, 103]], dtype=np.float64)
    result = torch_wrapper(a, b, alpha=alpha, c=c, beta=beta, device=device)
    assert np.allclose(
        result, expected
    ), f"{name} failed matrix multiplication with c and beta on {device}"
    print("  Passed matrix multiplication with c and beta")

    # Test case 4: Default values (alpha=1, c=None, beta=0)
    result = torch_wrapper(a, b, alpha=1, c=None, beta=0, device=device)
    assert np.allclose(
        result, np.array([[19, 22], [43, 50]])
    ), f"{name} failed with default values on {device}"
    print("  Passed matrix multiplication with default values")

    # Test case 5: Non-square matrices
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
    expected = np.array([[58, 64], [139, 154]], dtype=np.float64)
    result = torch_wrapper(a, b, device=device)
    assert np.allclose(
        result, expected
    ), f"{name} failed with non-square matrices on {device}"
    print("  Passed matrix multiplication with non-square matrices")

    print(f"{name} dot interface tests passed successfully on {device}!\n")


# Wrapper for PyTorch (CPU and GPU)
def torch_wrapper(a, b, alpha=1, c=None, beta=0, device="cpu"):
    a_torch = torch.tensor(a, device=device)
    b_torch = torch.tensor(b, device=device)
    if c is not None:
        c_torch = torch.tensor(c, device=device)
    else:
        c_torch = None
    result = torch_dot(a_torch, b_torch, alpha=alpha, c=c_torch, beta=beta)
    return result.cpu().numpy()


# PyTorch CPU test
test_dot_interface("PyTorch", device="cpu")

# PyTorch GPU test (if available)
if torch.cuda.is_available():
    test_dot_interface("PyTorch", device="cuda")
else:
    print("CUDA is not available. PyTorch GPU test skipped.")
