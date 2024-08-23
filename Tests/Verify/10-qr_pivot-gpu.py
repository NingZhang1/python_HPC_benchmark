import numpy
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._torch

numpy_qr = BackEnd._numpy.qr_col_pivoting
torch_qr = BackEnd._torch.qr_col_pivoting


def check_diagonal_descending(r):
    diag = numpy.abs(numpy.diag(r))
    return numpy.all(diag[:-1] >= diag[1:])


def test_qr_decomposition(sizes, device="cpu"):
    print(f"\nTesting on {device}")
    for m, n in sizes:
        print(f"Testing matrix of size {m}x{n}")

        # Create a random matrix
        np_matrix = numpy.random.rand(m, n)
        torch_matrix = torch.tensor(np_matrix, device=device)

        # Test numpy_qr
        print("Testing numpy_qr:")
        np_q, np_r, np_P = numpy_qr(np_matrix, mode="full")
        check_qr_result(np_matrix, np_q, np_r, np_P, "numpy_qr_col_pivoting", "cpu")

        # Test torch_qr
        print("Testing torch_qr:")
        torch_q, torch_r, torch_P = torch_qr(torch_matrix, mode="full")
        assert torch_q.device == torch_matrix.device
        assert torch_r.device == torch_matrix.device
        assert torch_P.device == torch_matrix.device
        check_qr_result(
            np_matrix,
            torch_q.cpu().numpy(),
            torch_r.cpu().numpy(),
            torch_P.cpu().numpy(),
            "torch_qr_col_pivoting",
            device,
        )


def check_qr_result(original, q, r, P, function_name, device):
    # Check shapes
    m, n = original.shape
    if q.shape != (m, m):
        print(f"  {function_name} error: Q shape is {q.shape}, expected ({m}, {m})")
    if r.shape != (m, n):
        print(f"  {function_name} error: R shape is {r.shape}, expected ({m}, {n})")

    # Check Q properties
    if not numpy.allclose(numpy.dot(q.T, q), numpy.eye(m), atol=1e-6):
        print(f"  {function_name} error: Q is not orthogonal")

    # Check R properties
    if not numpy.allclose(numpy.triu(r), r, atol=1e-6):
        print(f"  {function_name} error: R is not upper triangular")

    # Check if diagonal elements of R are in descending order by absolute value
    if check_diagonal_descending(r):
        print(
            f"  {function_name}: R diagonal elements are in descending order by absolute value"
        )
    else:
        print(
            f"  {function_name} error: R diagonal elements are not in descending order by absolute value"
        )

    # Check reconstruction
    reconstructed = numpy.dot(q, r)
    reconstructed2 = numpy.zeros_like(original)
    reconstructed2[:, P] = reconstructed.copy()

    if not numpy.allclose(original, reconstructed2, atol=1e-6):
        print(
            f"  {function_name} error: A ≠ QR, max difference: {numpy.max(numpy.abs(original - reconstructed2))}"
        )
    else:
        print(f"  {function_name}: QR decomposition is correct")

    # Check types and device
    if function_name.startswith("torch"):
        print(f"  {function_name}: Q type: {type(q).__name__}, device: {device}")
        print(f"  {function_name}: R type: {type(r).__name__}, device: {device}")
        print(f"  {function_name}: P type: {type(P).__name__}, device: {device}")


# Test with different matrix sizes, including non-square matrices
matrix_sizes = [(10, 10), (100, 100), (50, 75), (75, 50)]

if __name__ == "__main__":
    # CPU test
    test_qr_decomposition(matrix_sizes, "cpu")

    # GPU test
    if torch.cuda.is_available():
        test_qr_decomposition(matrix_sizes, "cuda")
    else:
        print("\nCUDA is not available. GPU test skipped.")
