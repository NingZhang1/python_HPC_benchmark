import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ij_j_ij = BackEnd._numpy.einsum_ij_j_ij
torch_einsum_ij_j_ij = BackEnd._torch.einsum_ij_j_ij


def run_test(device="cpu"):
    print(f"\nRunning test on {device}")

    m, n = 10, 12

    a = np.random.rand(m, n)
    b = np.random.rand(n)

    # NumPy test
    np_result = numpy_einsum_ij_j_ij(a, b)

    # PyTorch test
    a_torch = torch.tensor(a, device=device)
    b_torch = torch.tensor(b, device=device)
    torch_result = torch_einsum_ij_j_ij(a_torch, b_torch)
    torch_result_cpu = torch_result.cpu().numpy()

    # Benchmark
    benchmark = np.einsum("ij,j->ij", a, b)

    # Assertions
    assert np.allclose(np_result, benchmark), "NumPy result does not match benchmark"
    assert np.allclose(
        torch_result_cpu, benchmark
    ), f"PyTorch result on {device} does not match benchmark"

    print(f"All tests passed on {device}")


if __name__ == "__main__":
    # CPU test
    run_test("cpu")

    # GPU test
    if torch.cuda.is_available():
        run_test("cuda")
    else:
        print("\nCUDA is not available. GPU test skipped.")
