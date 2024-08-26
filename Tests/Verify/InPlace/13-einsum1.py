import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ij_j_ij = BackEnd._numpy.einsum_ij_j_ij
scipy_einsum_ij_j_ij = BackEnd._scipy.einsum_ij_j_ij
torch_einsum_ij_j_ij = BackEnd._torch.einsum_ij_j_ij


def test_einsum_ij_j_ij(operation, name, device="cpu"):
    # Create sample input tensors
    i, j = 3, 4
    a = np.random.rand(i, j)  # shape (i, j)
    b = np.random.rand(j)  # shape (j,)

    # Create an output array
    out = np.zeros((i, j))
    out_original = out.copy()

    if device == "cpu":
        result = operation(a, b, out=out)
    else:  # GPU
        a_gpu = torch.from_numpy(a).cuda()
        b_gpu = torch.from_numpy(b).cuda()
        out_gpu = torch.from_numpy(out).cuda()
        result_gpu = operation(a_gpu, b_gpu, out=out_gpu)
        assert out_gpu.data_ptr() == result_gpu.data_ptr()
        result = result_gpu.cpu().numpy()
        out = out_gpu.cpu().numpy()

    print(f"{name} einsum_ij_j_ij on {device}:")

    # Check if the output array was used
    if device == "cpu":
        is_out_used = np.may_share_memory(result, out)
        print(f"  {'Uses' if is_out_used else 'Does not use'} the provided out array.")

    # Check if the output array was modified
    is_out_modified = not np.array_equal(out, out_original)
    print(
        f"  {'Modifies' if is_out_modified else 'Does not modify'} the provided out array."
    )

    # Verify the shape of the result
    expected_shape = (i, j)
    shape_correct = result.shape == expected_shape
    print(
        f"  {'Produces' if shape_correct else 'Does not produce'} the correct shape {expected_shape}."
    )

    # Verify the result using numpy's einsum
    expected_result = np.einsum("ij,j->ij", a, b)
    is_correct = np.allclose(result, expected_result)
    print(f"  {'Produces' if is_correct else 'Does not produce'} the correct result.")

    # Check if the operation is element-wise multiplication
    manual_result = a * b[np.newaxis, :]
    is_elementwise = np.allclose(result, manual_result)
    print(
        f"  {'Is' if is_elementwise else 'Is not'} equivalent to element-wise multiplication."
    )
    print()


# Test NumPy and SciPy operations
for name, op in [("NumPy", numpy_einsum_ij_j_ij), ("SciPy", scipy_einsum_ij_j_ij)]:
    test_einsum_ij_j_ij(op, name)


# PyTorch CPU test
def torch_wrapper_cpu(operation):
    def wrapper(*args, **kwargs):
        torch_args = [
            torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        torch_kwargs = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        result = operation(*torch_args, **torch_kwargs)
        return result.numpy()

    return wrapper


test_einsum_ij_j_ij(torch_wrapper_cpu(torch_einsum_ij_j_ij), "PyTorch", "cpu")

# PyTorch GPU test
if torch.cuda.is_available():
    test_einsum_ij_j_ij(torch_einsum_ij_j_ij, "PyTorch", "gpu")
else:
    print("GPU is not available for PyTorch testing.")

# Test GPU availability
print("\nTesting GPU availability:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
