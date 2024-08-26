import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
# Note: BackEnd._config.disable_gpu() is not called as per the given import statements

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ik_jk_ijk = BackEnd._numpy.einsum_ik_jk_ijk
scipy_einsum_ik_jk_ijk = BackEnd._scipy.einsum_ik_jk_ijk
torch_einsum_ik_jk_ijk = BackEnd._torch.einsum_ik_jk_ijk


def test_einsum_ik_jk_ijk(operation, name, device="cpu"):
    # Create sample input tensors
    a = np.random.rand(3, 4)  # shape (i, k)
    b = np.random.rand(5, 4)  # shape (j, k)

    # Create an output array
    out = np.zeros((3, 5, 4))  # shape (i, j, k)

    if device == "cpu":
        result = operation(a, b, out=out)
    else:  # GPU
        a_gpu = torch.from_numpy(a).cuda()
        b_gpu = torch.from_numpy(b).cuda()
        out_gpu = torch.from_numpy(out).cuda()
        result_gpu = operation(a_gpu, b_gpu, out=out_gpu)
        result = result_gpu.cpu().numpy()

    # Check if the result is the same object as out
    is_out_used = (
        result.__array_interface__["data"][0] == out.__array_interface__["data"][0]
    )
    print(
        f"{name} einsum_ik_jk_ijk on {device} {'uses' if is_out_used else 'does not use'} the provided out array."
    )

    # Verify the shape of the result
    expected_shape = (3, 5, 4)
    shape_correct = result.shape == expected_shape
    print(
        f"{name} einsum_ik_jk_ijk on {device} {'produces' if shape_correct else 'does not produce'} the correct shape {expected_shape}."
    )

    # Verify the result using numpy's einsum
    expected_result = np.einsum("ik,jk->ijk", a, b)
    is_correct = np.allclose(result, expected_result)
    print(
        f"{name} einsum_ik_jk_ijk on {device} {'produces' if is_correct else 'does not produce'} the correct result."
    )
    print()


# Test NumPy and SciPy operations
for name, op in [("NumPy", numpy_einsum_ik_jk_ijk), ("SciPy", scipy_einsum_ik_jk_ijk)]:
    test_einsum_ik_jk_ijk(op, name)


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


test_einsum_ik_jk_ijk(torch_wrapper_cpu(torch_einsum_ik_jk_ijk), "PyTorch")

# PyTorch GPU test
if torch.cuda.is_available():
    test_einsum_ik_jk_ijk(torch_einsum_ik_jk_ijk, "PyTorch", device="gpu")
else:
    print("GPU is not available for PyTorch testing.")

# Test if GPU is actually disabled
print("\nTesting GPU availability:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
