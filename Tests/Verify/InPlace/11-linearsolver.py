import numpy as np
import scipy.linalg
import torch
import BackEnd._config

BackEnd._config.disable_fftw()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_solve_cholesky = BackEnd._numpy.solve_cholesky
scipy_solve_cholesky = BackEnd._scipy.solve_cholesky
torch_solve_cholesky = BackEnd._torch.solve_cholesky


def test_solve_cholesky(operation, name, device="cpu"):
    # Create a positive definite symmetric matrix A
    n = 5
    A = np.random.rand(n, n)
    A = np.dot(A, A.T) + np.eye(n)  # Ensure positive definiteness

    # Create a right-hand side matrix B with m columns
    m = 3
    B = np.random.rand(n, m)

    # Make copies of A and B for later comparison
    A_original = A.copy()
    B_original = B.copy()

    if device == "cpu":
        X = operation(
            A, B, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False
        )
    else:  # GPU
        A_gpu = torch.from_numpy(A).cuda()
        B_gpu = torch.from_numpy(B).cuda()
        X_gpu = operation(
            A_gpu,
            B_gpu,
            lower=True,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
        )
        X = X_gpu.cpu().numpy()
        A = A_gpu.cpu().numpy()
        B = B_gpu.cpu().numpy()

    # Check if A and B were overwritten
    is_A_overwritten = not np.array_equal(A, A_original)
    is_B_overwritten = not np.array_equal(B, B_original)
    print(f"{name} solve_cholesky on {device}:")
    print(
        f"  {'Overwrites' if is_A_overwritten else 'Does not overwrite'} the input matrix A."
    )
    print(
        f"  {'Overwrites' if is_B_overwritten else 'Does not overwrite'} the input matrix B."
    )

    # Verify the shape of the result
    shape_correct = X.shape == B_original.shape
    print(
        f"  {'Produces' if shape_correct else 'Does not produce'} the correct shape {B_original.shape}."
    )

    # Verify the result
    residual = np.linalg.norm(np.dot(A_original, X) - B_original)
    is_correct = residual < 1e-10
    print(
        f"  {'Produces' if is_correct else 'Does not produce'} a correct solution. Residual: {residual}"
    )
    print()


# Test NumPy and SciPy operations
for name, op in [("NumPy", numpy_solve_cholesky), ("SciPy", scipy_solve_cholesky)]:
    test_solve_cholesky(op, name)


# PyTorch CPU test
def torch_wrapper_cpu(operation):
    def wrapper(
        a, b, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False
    ):
        a_torch = torch.from_numpy(a)
        b_torch = torch.from_numpy(b)
        result = operation(
            a_torch,
            b_torch,
            lower=lower,
            overwrite_a=overwrite_a,
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        )
        return result.numpy()

    return wrapper


test_solve_cholesky(torch_wrapper_cpu(torch_solve_cholesky), "PyTorch")

# PyTorch GPU test
if torch.cuda.is_available():
    test_solve_cholesky(torch_solve_cholesky, "PyTorch", device="gpu")
else:
    print("GPU is not available for PyTorch testing.")

# Test if GPU is actually disabled
print("\nTesting GPU availability:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
