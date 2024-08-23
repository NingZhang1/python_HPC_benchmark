import numpy as np
import torch
import time
import timeit
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ik_jk_ijk = BackEnd._numpy.einsum_ik_jk_ijk
scipy_einsum_ik_jk_ijk = BackEnd._scipy.einsum_ik_jk_ijk
torch_einsum_ik_jk_ijk = BackEnd._torch.einsum_ik_jk_ijk


def time_function_cpu(func, *args, number=15):
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


def time_function_gpu(func, *args, number=15):
    # Warm-up run
    for _ in range(5):
        func(*args)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(number):
        func(*args)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

    return elapsed_time, elapsed_time


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
        ("NumPy", numpy_einsum_ik_jk_ijk, time_function_cpu),
        ("SciPy", scipy_einsum_ik_jk_ijk, time_function_cpu),
        (
            "PyTorch CPU",
            lambda A, B, out: torch_einsum_ik_jk_ijk(
                torch.from_numpy(A), torch.from_numpy(B), out=torch.from_numpy(out)
            ).numpy(),
            time_function_cpu,
        ),
        (
            "NumPy einsum",
            lambda A, B, out: np.einsum("ik,jk->ijk", A, B, out=out),
            time_function_cpu,
        ),
    ]

    if torch.cuda.is_available():
        implementations.append(
            (
                "PyTorch GPU",
                lambda A, B, out: torch_einsum_ik_jk_ijk(
                    torch.from_numpy(A).cuda(),
                    torch.from_numpy(B).cuda(),
                    out=torch.from_numpy(out).cuda(),
                )
                .cpu()
                .numpy(),
                time_function_gpu,
            )
        )

    for i, (A, B, C) in enumerate(test_cases):
        print(f"\nTest case {i+1}: A shape {A.shape}, B shape {B.shape}")

        for name, func, time_func in implementations:
            if name == "PyTorch GPU":
                gpu_time, _ = time_func(func, A, B, C)
                print(f"{name}:")
                print(f"  GPU time: {gpu_time:.6f} seconds")
            else:
                cpu_time, wall_time = time_func(func, A, B, C)
                print(f"{name}:")
                print(f"  CPU time: {cpu_time:.6f} seconds")
                print(f"  Wall time: {wall_time:.6f} seconds")


if __name__ == "__main__":
    run_performance_tests()
