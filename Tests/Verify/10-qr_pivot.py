import numpy
import torch

import backend._config

backend._config.disable_fftw()

import backend._numpy
import backend._scipy
import backend._torch

backend._torch.disable_gpu()

numpy_qr = backend._numpy.qr_col_pivoting
scipy_qr = backend._scipy.qr_col_pivoting
torch_qr = backend._torch.qr_col_pivoting


def check_diagonal_descending(r):
    diag = numpy.abs(numpy.diag(r))
    return numpy.all(diag[:-1] >= diag[1:])


def test_qr_decomposition(sizes):
    for m, n in sizes:
        print(f"Testing matrix of size {m}x{n}")

        # Create a random matrix
        np_matrix = numpy.random.rand(m, n)
        torch_matrix = torch.from_numpy(np_matrix)

        # Test numpy_qr
        print("Testing numpy_qr:")
        np_q, np_r, P = numpy_qr(np_matrix, mode="full")
        check_qr_result(np_matrix, np_q, np_r, P, "numpy_qr_col_pivoting")

        # Test scipy_qr
        print("Testing scipy_qr:")
        sp_q, sp_r, P = scipy_qr(np_matrix, mode="full")
        check_qr_result(np_matrix, sp_q, sp_r, P, "scipy_qr_col_pivoting")

        # Test torch_qr
        print("Testing torch_qr:")
        torch_q, torch_r, P = torch_qr(torch_matrix, mode="full")
        check_qr_result(
            np_matrix,
            torch_q.numpy(),
            torch_r.numpy(),
            P.numpy(),
            "torch_qr_col_pivoting",
        )


def check_qr_result(original, q, r, P, function_name):
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


# Test with different matrix sizes, including non-square matrices
matrix_sizes = [(10, 10), (100, 100), (50, 75), (75, 50)]
test_qr_decomposition(matrix_sizes)
