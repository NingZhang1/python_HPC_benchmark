import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
BackEnd._config.disable_gpu()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_qr = BackEnd._numpy.qr
scipy_qr = BackEnd._scipy.qr
torch_qr = BackEnd._torch.qr


def test_qr_decomposition(sizes):
    for m, n in sizes:
        print(f"Testing matrix of size {m}x{n}")

        # Create a random matrix
        np_matrix = numpy.random.rand(m, n)
        torch_matrix = torch.from_numpy(np_matrix)

        # Test numpy_qr
        print("Testing numpy_qr:")
        np_q, np_r = numpy_qr(np_matrix)
        check_qr_result(np_matrix, np_q, np_r, "numpy_qr")

        # Test scipy_qr
        print("Testing scipy_qr:")
        sp_q, sp_r = scipy_qr(np_matrix)
        check_qr_result(np_matrix, sp_q, sp_r, "scipy_qr")

        # Test torch_qr
        print("Testing torch_qr:")
        torch_q, torch_r = torch_qr(torch_matrix)
        check_qr_result(np_matrix, torch_q.numpy(), torch_r.numpy(), "torch_qr")


def check_qr_result(original, q, r, function_name):
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

    # Check reconstruction
    reconstructed = numpy.dot(q, r)
    if not numpy.allclose(original, reconstructed, atol=1e-6):
        print(
            f"  {function_name} error: A ≠ QR, max difference: {numpy.max(numpy.abs(original - reconstructed))}"
        )
    else:
        print(f"  {function_name}: QR decomposition is correct")


# Test with different matrix sizes, including non-square matrices
matrix_sizes = [(10, 10), (100, 100), (50, 75), (75, 50)]
test_qr_decomposition(matrix_sizes)
