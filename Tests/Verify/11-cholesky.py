import numpy
import scipy.linalg
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._scipy
import BackEnd._torch

numpy_cholesky = BackEnd._numpy.cholesky
scipy_cholesky = BackEnd._scipy.cholesky
torch_cholesky = BackEnd._torch.cholesky


def test_cholesky(sizes, device="cpu"):
    print(f"\nTesting on {device}")
    for n in sizes:
        print(f"Testing matrix of size {n}x{n}")

        # Create a random positive definite matrix
        A = numpy.random.rand(n, n)
        A = numpy.dot(A, A.T)  # Make it symmetric positive definite
        A2 = torch.from_numpy(A)  # no copy
        print("address of A : ", A.__array_interface__["data"][0])
        print("address of A2: ", A2.data_ptr())

        A_bench = A.copy()

        # NumPy test
        print("Testing numpy_cholesky:")
        A1 = A.copy()
        np_L = numpy_cholesky(A1)
        print("address of np_L : ", np_L.__array_interface__["data"], np_L.shape)
        print("address of A    : ", A.__array_interface__["data"], A1.shape)
        # assert np_L.__array_interface__["data"][0] == A1.__array_interface__["data"][0]
        check_cholesky(A, np_L, "numpy_cholesky", "cpu")

        # SciPy test
        print("Testing scipy_cholesky:")
        A2 = A.copy()
        sp_L = scipy_cholesky(A2)
        print("address of np_L : ", np_L.__array_interface__["data"])
        print("address of A    : ", A2.__array_interface__["data"])
        # assert sp_L.__array_interface__["data"][0] == A2.__array_interface__["data"][0]
        check_cholesky(A, sp_L, "scipy_cholesky", "cpu")

        # PyTorch test
        print("Testing torch_cholesky:")
        A_torch = torch.tensor(A, device=device)  # copy
        print("address of A_torch : ", A_torch.data_ptr())
        print("address of A       : ", A.__array_interface__["data"][0])
        torch_L = torch_cholesky(A_torch)
        assert torch_L.device == A_torch.device
        # assert torch_L.data_ptr() != A_torch.data_ptr()
        print("address of torch_L: ", torch_L.data_ptr())
        # check inplace #
        print("address of A_torch: ", A_torch.data_ptr())
        check_cholesky(A_bench, torch_L.cpu().numpy(), "torch_cholesky", device)


def check_cholesky(A, L, function_name, device):
    # Check if L is lower triangular
    if not numpy.allclose(L, numpy.tril(L)):
        print(f"  {function_name} error: L is not lower triangular")

    # Check if L * L^T = A
    reconstructed_A = numpy.dot(L, L.T)
    if numpy.allclose(A, reconstructed_A):
        print(f"  {function_name}: Cholesky decomposition is correct")
    else:
        print(f"  {function_name} error: L * L^T does not equal A")
        print(f"    Max difference: {numpy.max(numpy.abs(A - reconstructed_A))}")

    # Check types and device
    if function_name.startswith("torch"):
        print(f"  {function_name}: L type: {type(L).__name__}, device: {device}")

    # Check shapes
    if L.shape != A.shape:
        print(f"  {function_name} error: L shape is {L.shape}, expected {A.shape}")


# Test with different matrix sizes
matrix_sizes = [10, 100, 500]

if __name__ == "__main__":
    # CPU test
    test_cholesky(matrix_sizes, "cpu")

    # GPU test
    if torch.cuda.is_available():
        test_cholesky(matrix_sizes, "cuda")
    else:
        print("\nCUDA is not available. GPU test skipped.")
