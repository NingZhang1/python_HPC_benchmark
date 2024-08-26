import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
# BackEnd._config.disable_gpu()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_index_add = BackEnd._numpy.index_add
scipy_index_add = BackEnd._scipy.index_add
torch_index_add = BackEnd._torch.index_add

numpy_index_copy = BackEnd._numpy.index_copy
scipy_index_copy = BackEnd._scipy.index_copy
torch_index_copy = BackEnd._torch.index_copy

numpy_take = BackEnd._numpy.take
scipy_take = BackEnd._scipy.take
torch_take = BackEnd._torch.take


def test_inplace_operation(operation, name, op_type):
    if op_type in ["index_add", "index_copy"]:
        A = np.random.rand(5, 5)
        dim = 0
        index = np.array([0, 2, 4])
        source = np.random.rand(3, 5)
        if op_type == "index_add":
            result = operation(A, dim, index, source, alpha=1)
        else:  # index_copy
            result = operation(A, dim, index, source)

        # Check if the result shares memory with the input
        is_inplace = (
            result.__array_interface__["data"][0] == A.__array_interface__["data"][0]
        )
        print(
            f"{name} {op_type} is {'an' if is_inplace else 'not an'} inplace operation."
        )

    elif op_type == "take":
        a = np.random.rand(5, 5)
        indices = np.array([0, 2, 4])
        out = np.zeros((3, 5))
        result = operation(a, indices, axis=0, out=out)

        # Check if the result is the same object as out
        is_out_used = (
            result.__array_interface__["data"][0] == out.__array_interface__["data"][0]
        )
        print(
            f"{name} {op_type} {'uses' if is_out_used else 'does not use'} the provided out array."
        )


# Test NumPy and SciPy operations
for op_type, np_op, sp_op in [
    ("index_add", numpy_index_add, scipy_index_add),
    ("index_copy", numpy_index_copy, scipy_index_copy),
    ("take", numpy_take, scipy_take),
]:
    test_inplace_operation(np_op, "NumPy", op_type)
    test_inplace_operation(sp_op, "SciPy", op_type)


# PyTorch CPU test
def torch_wrapper_cpu(operation, op_type):
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


for op_type, torch_op in [
    ("index_add", torch_index_add),
    ("index_copy", torch_index_copy),
    ("take", torch_take),
]:
    test_inplace_operation(
        torch_wrapper_cpu(torch_op, op_type), "PyTorch (CPU)", op_type
    )

# PyTorch GPU test
if torch.cuda.is_available():

    def torch_wrapper_gpu(operation, op_type):
        def wrapper(*args, **kwargs):
            torch_args = [
                (
                    arg.cuda()
                    if isinstance(arg, torch.Tensor)
                    else (
                        torch.from_numpy(arg).cuda()
                        if isinstance(arg, np.ndarray)
                        else arg
                    )
                )
                for arg in args
            ]
            torch_kwargs = {
                k: (
                    v.cuda()
                    if isinstance(v, torch.Tensor)
                    else torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v
                )
                for k, v in kwargs.items()
            }
            result = operation(*torch_args, **torch_kwargs)
            return result.cpu().numpy()

        return wrapper

    for op_type, torch_op in [
        ("index_add", torch_index_add),
        ("index_copy", torch_index_copy),
        ("take", torch_take),
    ]:
        test_inplace_operation(
            torch_wrapper_gpu(torch_op, op_type), "PyTorch (GPU)", op_type
        )
else:
    print("GPU is not available for PyTorch testing.")

# Test if GPU is actually disabled
print("\nTesting if GPU is actually disabled:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
