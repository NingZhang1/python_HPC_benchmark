import numpy
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._torch

numpy_qr = BackEnd._numpy.qr
torch_qr = BackEnd._torch.qr


def test_qr_decomposition(sizes, device="cpu"):
    print(f"\nTesting on {device}")
    for m, n in sizes:
        print(f"Testing matrix of size {m}x{n}")

        # Create a random matrix
        np_matrix = numpy.random.rand(m, n)
        torch_matrix = torch.tensor(np_matrix, device=device)

        # Test numpy_qr
        print("Testing numpy_qr:")
        np_q, np_r = numpy_qr(np_matrix)
        check_qr_result(np_matrix, np_q, np_r, "numpy_qr", "cpu")

        # Test torch_qr
        print("Testing torch_qr:")
        torch_q, torch_r = torch_qr(torch_matrix)
        assert torch_q.device == torch_matrix.device
        assert torch_r.device == torch_matrix.device
        check_qr_result(
            np_matrix, torch_q.cpu().numpy(), torch_r.cpu().numpy(), "torch_qr", device
        )


def check_qr_result(original, q, r, function_name, device):
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

    # Check types and device
    if function_name.startswith("torch"):
        print(f"  {function_name}: Q type: {type(q).__name__}, device: {device}")
        print(f"  {function_name}: R type: {type(r).__name__}, device: {device}")


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
