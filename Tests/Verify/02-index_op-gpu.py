import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._torch

numpy_index_add = BackEnd._numpy.index_add
numpy_index_copy = BackEnd._numpy.index_copy
numpy_take = BackEnd._numpy.take

torch_index_add = BackEnd._torch.index_add
torch_index_copy = BackEnd._torch.index_copy
torch_take = BackEnd._torch.take


def verify_index_add(A_np, dim, index, source, alpha=1, device="cpu"):
    A_numpy = A_np.copy()
    A_torch = torch.tensor(A_np, device=device)
    source_torch = torch.tensor(source, device=device)
    index_torch = torch.tensor(index, device=device)

    numpy_index_add(A_numpy, dim, index, source, alpha=alpha)
    torch_index_add(A_torch, dim, index_torch, source_torch, alpha=alpha)

    assert np.allclose(
        A_numpy, A_torch.cpu().numpy()
    ), f"NumPy and PyTorch index_add results do not match on {device}"
    print(f"index_add verification passed on {device}")


def verify_index_copy(A_np, dim, index, source, device="cpu"):
    A_numpy = A_np.copy()
    A_torch = torch.tensor(A_np, device=device)
    source_torch = torch.tensor(source, device=device)
    index_torch = torch.tensor(index, device=device)

    numpy_index_copy(A_numpy, dim, index, source)
    torch_index_copy(A_torch, dim, index_torch, source_torch)

    assert np.allclose(
        A_numpy, A_torch.cpu().numpy()
    ), f"NumPy and PyTorch index_copy results do not match on {device}"
    print(f"index_copy verification passed on {device}")


def verify_take(a_np, indices, axis=None, device="cpu"):
    a_torch = torch.tensor(a_np, device=device)
    indices_torch = torch.tensor(indices, device=device)

    result_numpy = numpy_take(a_np, indices, axis=axis)
    result_torch = torch_take(a_torch, indices_torch, axis=axis).cpu().numpy()

    assert np.allclose(
        result_numpy, result_torch
    ), f"NumPy and PyTorch take results do not match on {device}"
    print(f"take verification passed on {device}")


def run_verifications(device="cpu"):
    print(f"Running verifications on {device}")

    # Test case 1: 2D array, add to rows
    A = np.random.rand(5, 3)
    index = np.array([1, 3])
    source = np.random.rand(2, 3)
    verify_index_add(A, 0, index, source, device=device)
    verify_index_copy(A, 0, index, source, device=device)

    # Test case 2: 2D array, add to columns
    A = np.random.rand(3, 5)
    index = np.array([0, 2, 4])
    source = np.random.rand(3, 3)
    verify_index_add(A, 1, index, source, device=device)
    verify_index_copy(A, 1, index, source, device=device)

    # Test case 3: 3D array
    A = np.random.rand(4, 3, 5)
    index = np.array([1, 2])
    source = np.random.rand(2, 3, 5)
    verify_index_add(A, 0, index, source, device=device)
    verify_index_copy(A, 0, index, source, device=device)

    # Test case 4: with alpha
    A = np.random.rand(5, 3)
    index = np.array([1, 3])
    source = np.random.rand(2, 3)
    alpha = 0.5
    verify_index_add(A, 0, index, source, alpha, device=device)

    # Test case 5: take function
    a = np.random.rand(10, 5)
    indices = np.array([1, 3, 4])
    verify_take(a, indices, axis=0, device=device)
    verify_take(a, indices, axis=1, device=device)

    print(f"All verifications passed successfully on {device}!")


if __name__ == "__main__":
    run_verifications("cpu")
    if torch.cuda.is_available():
        run_verifications("cuda")
    else:
        print("CUDA is not available. GPU tests skipped.")
