import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ik_jk_ijk = BackEnd._numpy.einsum_ik_jk_ijk
torch_einsum_ik_jk_ijk = BackEnd._torch.einsum_ik_jk_ijk


def run_test(device="cpu"):
    print(f"\nRunning test on {device}")

    m, n, k = 11, 13, 17

    a = np.random.rand(m, k)
    b = np.random.rand(n, k)

    # NumPy test
    np_result = numpy_einsum_ik_jk_ijk(a, b)

    # PyTorch test
    a_torch = torch.tensor(a, device=device)
    b_torch = torch.tensor(b, device=device)
    torch_result = torch_einsum_ik_jk_ijk(a_torch, b_torch)
    torch_result_cpu = torch_result.cpu().numpy()

    # Benchmark
    benchmark = np.einsum("ik,jk->ijk", a, b)

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
