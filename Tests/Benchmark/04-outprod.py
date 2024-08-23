import numpy as np
import torch
import time
import timeit
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._scipy
import BackEnd._torch

BackEnd._torch.disable_gpu()

numpy_einsum_ik_jk_ijk = BackEnd._numpy.einsum_ik_jk_ijk
scipy_einsum_ik_jk_ijk = BackEnd._scipy.einsum_ik_jk_ijk
torch_einsum_ik_jk_ijk = BackEnd._torch.einsum_ik_jk_ijk


def time_function(func, *args, number=15):
    # Warm-up run
    func(*args)

    # CPU time
    cpu_start = time.process_time()
    timeit.timeit(lambda: func(*args), number=number)
    cpu_time = time.process_time() - cpu_start

    # Wall time
    wall_start = time.time()
    timeit.timeit(lambda: func(*args), number=number)
    wall_time = time.time() - wall_start

    return cpu_time, wall_time


def run_performance_tests():
    # Test cases
    test_cases = [
        (
            np.random.rand(10, 20),
            np.random.rand(15, 20),
            np.zeros((10, 15, 20)),
        ),  # Small
        (
            np.random.rand(100, 200),
            np.random.rand(150, 200),
            np.zeros((100, 150, 200)),
        ),  # Medium
        (
            np.random.rand(500, 1000),
            np.random.rand(750, 1000),
            np.zeros((500, 750, 1000)),
        ),  # Large
    ]

    implementations = [
        ("NumPy", numpy_einsum_ik_jk_ijk),
        ("SciPy", scipy_einsum_ik_jk_ijk),
        (
            "PyTorch",
            lambda A, B, out: torch_einsum_ik_jk_ijk(
                torch.from_numpy(A), torch.from_numpy(B), out=torch.from_numpy(out)
            ).numpy(),
        ),
        ("NumPy einsum", lambda A, B, out: np.einsum("ik,jk->ijk", A, B, out=out)),
    ]

    for i, (A, B, C) in enumerate(test_cases):
        print(f"\nTest case {i+1}: A shape {A.shape}, B shape {B.shape}")

        for name, func in implementations:
            cpu_time, wall_time = time_function(func, A, B, C)
            print(f"{name}:")
            print(f"  CPU time: {cpu_time:.6f} seconds")
            print(f"  Wall time: {wall_time:.6f} seconds")


if __name__ == "__main__":
    run_performance_tests()
