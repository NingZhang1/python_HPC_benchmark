import numpy as np
import torch
import BackEnd._config
import time

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_i_ij_ij = BackEnd._numpy.einsum_i_ij_ij
scipy_einsum_i_ij_ij = BackEnd._scipy.einsum_i_ij_ij
torch_einsum_i_ij_ij = BackEnd._torch.einsum_i_ij_ij


def generate_test_cases():
    """Generate test cases of different sizes"""
    sizes = [
        (10, 5),  # Small
        (100, 50),  # Medium
        (1000, 500),  # Large
        (5000, 2000),  # Very large
        (10000, 5000),  # Extreme
    ]
    test_cases = []
    for i, j in sizes:
        a_np = np.random.rand(i)
        b_np = np.random.rand(i, j)
        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        out_np = np.empty((i, j))
        out_torch = torch.empty((i, j))
        out_torch_gpu = (
            torch.empty((i, j), device="cuda") if torch.cuda.is_available() else None
        )
        test_cases.append(
            (a_np, b_np, a_torch, b_torch, out_np, out_torch, out_torch_gpu, f"{i}x{j}")
        )
    return test_cases


def measure_time_cpu(func, *args, num_runs=100):
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


def measure_time_gpu(func, *args, num_runs=100):
    """Measure GPU time for a function"""
    # Warm-up run
    for _ in range(5):
        func(*args)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_runs):
        func(*args)
    end_event.record()

    torch.cuda.synchronize()
    gpu_time = start_event.elapsed_time(end_event) / (
        num_runs * 1000
    )  # Convert to seconds

    return gpu_time, gpu_time


def run_performance_test(test_cases, num_runs=100):
    """Run performance test for each implementation"""
    implementations = [
        (
            "NumPy Einsum",
            lambda a, b, out: numpy_einsum_i_ij_ij(a, b, out=out),
            measure_time_cpu,
        ),
        (
            "SciPy Einsum",
            lambda a, b, out: scipy_einsum_i_ij_ij(a, b, out=out),
            measure_time_cpu,
        ),
        (
            "PyTorch Einsum (CPU)",
            lambda a, b, out: torch_einsum_i_ij_ij(a, b, out=out),
            measure_time_cpu,
        ),
    ]

    if torch.cuda.is_available():
        implementations.append(
            (
                "PyTorch Einsum (GPU)",
                lambda a, b, out: torch_einsum_i_ij_ij(a, b, out=out),
                measure_time_gpu,
            )
        )

    results = {}

    for (
        a_np,
        b_np,
        a_torch,
        b_torch,
        out_np,
        out_torch,
        out_torch_gpu,
        size,
    ) in test_cases:
        print(f"\nPerformance test for size {size}:")
        print(
            f"{'Implementation':<25} {'CPU/GPU Time (ms)':<20} {'Wall Time (ms)':<20}"
        )
        print("-" * 65)

        size_results = {}

        for name, func, measure_func in implementations:
            if name == "PyTorch Einsum (GPU)":
                a, b, out = a_torch.cuda(), b_torch.cuda(), out_torch_gpu
            elif "PyTorch" in name:
                a, b, out = a_torch, b_torch, out_torch
            else:
                a, b, out = a_np, b_np, out_np

            time1, time2 = measure_func(func, a, b, out, num_runs=num_runs)
            time1_ms = time1 * 1000
            time2_ms = time2 * 1000

            print(f"{name:<25} {time1_ms:<20.6f} {time2_ms:<20.6f}")

            size_results[name] = (time1_ms, time2_ms)

        results[size] = size_results

    return results


# Run the performance test
test_cases = generate_test_cases()
performance_results = run_performance_test(test_cases)

# Print summary
print("\nPerformance Summary:")
headers = [
    "Size",
    "NumPy CPU",
    "NumPy Wall",
    "SciPy CPU",
    "SciPy Wall",
    "PyTorch CPU",
    "PyTorch Wall",
]
if torch.cuda.is_available():
    headers.extend(["PyTorch GPU", "PyTorch GPU"])

print(" ".join(f"{h:<15}" for h in headers))
print("-" * (15 * len(headers)))

for size, size_results in performance_results.items():
    numpy_cpu, numpy_wall = size_results["NumPy Einsum"]
    scipy_cpu, scipy_wall = size_results["SciPy Einsum"]
    pytorch_cpu, pytorch_wall = size_results["PyTorch Einsum (CPU)"]
    row = [
        f"{size:<15}",
        f"{numpy_cpu:<15.6f}",
        f"{numpy_wall:<15.6f}",
        f"{scipy_cpu:<15.6f}",
        f"{scipy_wall:<15.6f}",
        f"{pytorch_cpu:<15.6f}",
        f"{pytorch_wall:<15.6f}",
    ]
    if torch.cuda.is_available():
        pytorch_gpu, _ = size_results["PyTorch Einsum (GPU)"]
        row.extend([f"{pytorch_gpu:<15.6f}", f"{pytorch_gpu:<15.6f}"])
    print("".join(row))
