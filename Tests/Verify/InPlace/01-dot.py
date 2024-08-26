import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
# BackEnd._config.disable_gpu()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_dot = BackEnd._numpy.dot
scipy_dot = BackEnd._scipy.dot
torch_dot = BackEnd._torch.dot


def test_inplace_dot(dot_function, name):
    # Create sample matrices
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)
    c = np.random.rand(3, 3)

    # Perform dot product
    result = dot_function(a, b, c=c)

    # Check if the result has the same memory address as c
    is_inplace = (
        result.__array_interface__["data"][0] == c.__array_interface__["data"][0]
    )

    print(
        f"{name} dot product is {'an' if is_inplace else 'not an'} inplace operation."
    )


# Test NumPy and SciPy dot product functions
test_inplace_dot(numpy_dot, "NumPy")
test_inplace_dot(scipy_dot, "SciPy")


# PyTorch CPU test
def torch_dot_wrapper_cpu(a, b, c=None):
    a_torch = torch.from_numpy(a)
    b_torch = torch.from_numpy(b)
    c_torch = torch.from_numpy(c) if c is not None else None
    result = torch_dot(a_torch, b_torch, c=c_torch)
    return result.numpy()


test_inplace_dot(torch_dot_wrapper_cpu, "PyTorch (CPU)")

# PyTorch GPU test
if torch.cuda.is_available():

    def torch_dot_wrapper_gpu(a, b, c=None):
        a_torch = torch.from_numpy(a).cuda()
        b_torch = torch.from_numpy(b).cuda()
        c_torch = torch.from_numpy(c).cuda() if c is not None else None
        result = torch_dot(a_torch, b_torch, c=c_torch)
        return result.cpu().numpy()

    test_inplace_dot(torch_dot_wrapper_gpu, "PyTorch (GPU)")
else:
    print("GPU is not available for PyTorch testing.")

# Test if BackEnd._config.disable_gpu() actually disabled GPU
print("\nTesting if GPU is actually disabled:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
