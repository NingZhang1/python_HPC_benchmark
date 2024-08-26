import numpy as np
import scipy.linalg
import torch
import BackEnd._config

BackEnd._config.disable_fftw()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_cholesky = BackEnd._numpy.cholesky
scipy_cholesky = BackEnd._scipy.cholesky
torch_cholesky = BackEnd._torch.cholesky


def test_cholesky(operation, name, device="cpu"):
    # Create a positive definite symmetric matrix
    n = 5
    A = np.random.rand(n, n)
    A = np.dot(A, A.T) + np.eye(n)  # Ensure positive definiteness

    # Make a copy of A for later comparison
    A_original = A.copy()

    if device == "cpu":
        result = operation(A, lower=True, overwrite_a=True, out=None)
        if isinstance(result, np.ndarray):
            data_A = A.__array_interface__["data"][0]
            data_result = result.__array_interface__["data"][0]
            print("check if the result is in-place %s, %s" % (data_A, data_result))
    else:  # GPU
        A_gpu = torch.from_numpy(A).cuda()
        result_gpu = operation(A_gpu, lower=True, overwrite_a=True, out=None)
        result = result_gpu.cpu().numpy()
        A = A_gpu.cpu().numpy()  # Get the potentially modified A back to CPU

    # Check if A was overwritten
    is_overwritten = not np.array_equal(A, A_original)
    print(
        f"{name} Cholesky on {device} {'overwrites' if is_overwritten else 'does not overwrite'} the input matrix."
    )

    # Verify the shape of the result
    shape_correct = result.shape == A_original.shape
    print(
        f"{name} Cholesky on {device} {'produces' if shape_correct else 'does not produce'} the correct shape {A_original.shape}."
    )

    # Verify the result
    reconstructed = np.dot(result, result.T)
    is_correct = np.allclose(reconstructed, A_original)
    print(
        f"{name} Cholesky on {device} {'produces' if is_correct else 'does not produce'} a correct decomposition."
    )

    # Check if the result is lower triangular
    is_lower = np.allclose(result, np.tril(result))
    print(
        f"{name} Cholesky on {device} {'produces' if is_lower else 'does not produce'} a lower triangular matrix."
    )
    print()


# Test NumPy and SciPy operations
for name, op in [("NumPy", numpy_cholesky), ("SciPy", scipy_cholesky)]:
    test_cholesky(op, name)


# PyTorch CPU test
def torch_wrapper_cpu(operation):
    def wrapper(a, lower=True, overwrite_a=True, out=None):
        assert out is None
        a_torch = torch.from_numpy(a)
        result = operation(a_torch, lower=lower, overwrite_a=overwrite_a, out=out)
        assert a_torch.data_ptr() == result.data_ptr(), "Must be in-place."
        return result.numpy()

    return wrapper


test_cholesky(torch_wrapper_cpu(torch_cholesky), "PyTorch")

# PyTorch GPU test
if torch.cuda.is_available():
    test_cholesky(torch_cholesky, "PyTorch", device="gpu")
else:
    print("GPU is not available for PyTorch testing.")

# Test if GPU is actually disabled
print("\nTesting GPU availability:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
