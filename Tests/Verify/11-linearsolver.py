import numpy
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


def test_solve_cholesky(sizes, device="cpu"):
    print(f"\nTesting on {device}")
    for n, m in sizes:
        print(f"Testing matrix A of size {n}x{n} and B of size {n}x{m}")

        # Create a random positive definite matrix A and a random matrix B
        A = numpy.random.rand(n, n)
        A = numpy.dot(A, A.T)  # Make it symmetric positive definite
        B = numpy.random.rand(n, m)

        # NumPy test
        print("Testing numpy_solve_cholesky:")
        A1 = A.copy()
        B1 = B.copy()
        np_X = numpy_solve_cholesky(A1, B1)
        # assert np_X.__array_interface__["data"][0] == B.__array_interface__["data"][0]
        check_solution(A, B, np_X, "numpy_solve_cholesky", "cpu")

        # SciPy test
        print("Testing scipy_solve_cholesky:")
        A2 = A.copy()
        B2 = B.copy()
        sp_X = scipy_solve_cholesky(A2, B2)
        # assert sp_X.__array_interface__["data"][0] == B.__array_interface__["data"][0]
        check_solution(A, B, sp_X, "scipy_solve_cholesky", "cpu")

        # PyTorch test
        print("Testing torch_solve_cholesky:")
        A_torch = torch.tensor(A, device=device)
        B_torch = torch.tensor(B, device=device)
        torch_X = torch_solve_cholesky(A_torch, B_torch)
        assert torch_X.device == A_torch.device
        assert torch_X.data_ptr() == B_torch.data_ptr()
        check_solution(A, B, torch_X.cpu().numpy(), "torch_solve_cholesky", device)


def check_solution(A, B, X, function_name, device):
    # Check solution
    residual = numpy.linalg.norm(numpy.dot(A, X) - B)
    if residual < 1e-6:
        print(f"  {function_name}: Solution is correct. Residual: {residual}")
    else:
        print(f"  {function_name} error: Solution is incorrect. Residual: {residual}")

    # Check types and device
    if function_name.startswith("torch"):
        print(f"  {function_name}: X type: {type(X).__name__}, device: {device}")

    # Check shapes
    if X.shape != (A.shape[1], B.shape[1]):
        print(
            f"  {function_name} error: X shape is {X.shape}, expected {(A.shape[1], B.shape[1])}"
        )


# Test with different matrix sizes
matrix_sizes = [(10, 12), (100, 500), (500, 1200)]

if __name__ == "__main__":
    # CPU test
    test_solve_cholesky(matrix_sizes, "cpu")

    # GPU test
    if torch.cuda.is_available():
        test_solve_cholesky(matrix_sizes, "cuda")
    else:
        print("\nCUDA is not available. GPU test skipped.")
