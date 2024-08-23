import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._scipy
import BackEnd._torch

BackEnd._torch.disable_gpu()
import time

numpy_einsum_ij_j_ij = BackEnd._numpy.einsum_ij_j_ij
scipy_einsum_ij_j_ij = BackEnd._scipy.einsum_ij_j_ij
torch_einsum_ij_j_ij = BackEnd._torch.einsum_ij_j_ij


def generate_test_cases():
    """Generate test cases of different sizes"""
    sizes = [
        (10, 5),  # Small
        (100, 50),  # Medium
        (1000, 500),  # Large
        (5000, 2000),  # Very large
        (2000, 20000),
    ]
    test_cases = []
    for i, j in sizes:
        a_np = np.random.rand(i, j)
        b_np = np.random.rand(j)
        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        test_cases.append((a_np, b_np, a_torch, b_torch, f"{i}x{j}"))
    return test_cases


def measure_time(func, *args, num_runs=100):
    """Measure both CPU time and wall time for a function"""
    # Warm-up run
    func(*args)

    # Timed runs
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    for _ in range(num_runs):
        func(*args)

    cpu_end = time.process_time()
    wall_end = time.perf_counter()

    cpu_time = (cpu_end - cpu_start) / num_runs
    wall_time = (wall_end - wall_start) / num_runs

    return cpu_time, wall_time


def run_performance_test(test_cases, num_runs=100):
    """Run performance test for each implementation"""
    implementations = [
        ("NumPy Einsum", lambda a, b: numpy_einsum_ij_j_ij(a, b)),
        ("SciPy Einsum", lambda a, b: scipy_einsum_ij_j_ij(a, b)),
        ("PyTorch Einsum", lambda a, b: torch_einsum_ij_j_ij(a, b)),
    ]

    results = {}

    for a_np, b_np, a_torch, b_torch, size in test_cases:
        print(f"\nPerformance test for size {size}:")
        print(f"{'Implementation':<20} {'CPU Time (ms)':<15} {'Wall Time (ms)':<15}")
        print("-" * 50)

        size_results = {}

        for name, func in implementations:
            if name == "PyTorch Einsum":
                a, b = a_torch, b_torch
            else:
                a, b = a_np, b_np

            cpu_time, wall_time = measure_time(func, a, b, num_runs=num_runs)
            cpu_time_ms = cpu_time * 1000
            wall_time_ms = wall_time * 1000

            print(f"{name:<20} {cpu_time_ms:<15.6f} {wall_time_ms:<15.6f}")

            size_results[name] = (cpu_time_ms, wall_time_ms)

        results[size] = size_results

    return results


# Run the performance test
test_cases = generate_test_cases()
performance_results = run_performance_test(test_cases)

# Print summary
print("\nPerformance Summary:")
print(
    f"{'Size':<10} {'NumPy CPU':<12} {'NumPy Wall':<12} {'SciPy CPU':<12} {'SciPy Wall':<12} {'PyTorch CPU':<12} {'PyTorch Wall':<12}"
)
print("-" * 82)
for size, size_results in performance_results.items():
    numpy_cpu, numpy_wall = size_results["NumPy Einsum"]
    scipy_cpu, scipy_wall = size_results["SciPy Einsum"]
    pytorch_cpu, pytorch_wall = size_results["PyTorch Einsum"]
    print(
        f"{size:<10} {numpy_cpu:<12.6f} {numpy_wall:<12.6f} {scipy_cpu:<12.6f} {scipy_wall:<12.6f} {pytorch_cpu:<12.6f} {pytorch_wall:<12.6f}"
    )
