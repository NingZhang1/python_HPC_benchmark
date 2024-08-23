import numpy
import scipy.linalg
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_eigh = BackEnd._numpy.eigh
scipy_eigh = BackEnd._scipy.eigh
torch_eigh = BackEnd._torch.eigh


def test_eigh(sizes, device="cpu"):
    print(f"\nTesting on {device}")
    for n in sizes:
        print(f"Testing matrix of size {n}x{n}")

        # Create a random Hermitian matrix
        A = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
        A = A + A.conj().T

        # NumPy test
        print("Testing numpy_eigh:")
        np_w, np_v = numpy_eigh(A)
        check_eigh(A, np_w, np_v, "numpy_eigh", "cpu")

        # SciPy test
        print("Testing scipy_eigh:")
        sp_w, sp_v = scipy_eigh(A)
        check_eigh(A, sp_w, sp_v, "scipy_eigh", "cpu")

        # PyTorch test
        print("Testing torch_eigh:")
        A_torch = torch.tensor(A, device=device)
        torch_w, torch_v = torch_eigh(A_torch)
        check_eigh(
            A, torch_w.cpu().numpy(), torch_v.cpu().numpy(), "torch_eigh", device
        )


def check_eigh(A, w, v, function_name, device):
    # Check if eigenvalues are real
    if not numpy.allclose(w.imag, 0, atol=1e-7):
        print(f"  {function_name} error: Eigenvalues are not real")

    # Check if eigenvalues are in ascending order
    if not numpy.all(numpy.diff(w) >= 0):
        print(f"  {function_name} error: Eigenvalues are not in ascending order")

    # Check orthogonality of eigenvectors
    if not numpy.allclose(numpy.dot(v.conj().T, v), numpy.eye(len(w)), atol=1e-7):
        print(f"  {function_name} error: Eigenvectors are not orthogonal")

    # Check Av = λv
    for i in range(len(w)):
        if not numpy.allclose(numpy.dot(A, v[:, i]), w[i] * v[:, i], atol=1e-7):
            print(f"  {function_name} error: Av ≠ λv for eigenpair {i}")
            break
    else:
        print(f"  {function_name}: All eigenpairs satisfy Av = λv")

    # Check types and device
    if function_name.startswith("torch"):
        print(
            f"  {function_name}: w type: {type(w).__name__}, v type: {type(v).__name__}, device: {device}"
        )

    # Check shapes
    if w.shape != (A.shape[0],):
        print(
            f"  {function_name} error: w shape is {w.shape}, expected {(A.shape[0],)}"
        )
    if v.shape != A.shape:
        print(f"  {function_name} error: v shape is {v.shape}, expected {A.shape}")


# Test with different matrix sizes
matrix_sizes = [10, 50, 100]

if __name__ == "__main__":
    # CPU test
    test_eigh(matrix_sizes, "cpu")

    # GPU test
    if torch.cuda.is_available():
        test_eigh(matrix_sizes, "cuda")
    else:
        print("\nCUDA is not available. GPU test skipped.")
