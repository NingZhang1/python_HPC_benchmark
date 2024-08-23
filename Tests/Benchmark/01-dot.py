import numpy
import torch
import time
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_dot = BackEnd._numpy.dot
scipy_dot = BackEnd._scipy.dot
torch_dot = BackEnd._torch.dot


def measure_time_cpu(func, *args):
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    result = func(*args)

    end_cpu = time.process_time()
    end_wall = time.perf_counter()

    return result, end_cpu - start_cpu, end_wall - start_wall


def measure_time_gpu(func, *args):
    # Warm-up
    for _ in range(5):
        func(*args)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    result = func(*args)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

    return result, elapsed_time, elapsed_time


def test_dot_performance(sizes, num_runs=10):
    for m, n, k in sizes:
        print(f"\nTesting performance for matrix sizes: ({m},{n}) @ ({n},{k})")

        # Create random matrices
        a = numpy.random.rand(m, n)
        b = numpy.random.rand(n, k)
        torch_a = torch.from_numpy(a)
        torch_b = torch.from_numpy(b)

        # Test numpy_dot
        numpy_cpu_times = []
        numpy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(numpy_dot, a, b)
            numpy_cpu_times.append(cpu_time)
            numpy_wall_times.append(wall_time)

        print(
            f"numpy_dot: Avg CPU time: {numpy.mean(numpy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(numpy_wall_times):.6f}s"
        )

        # Test scipy_dot
        scipy_cpu_times = []
        scipy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(scipy_dot, a, b)
            scipy_cpu_times.append(cpu_time)
            scipy_wall_times.append(wall_time)

        print(
            f"scipy_dot: Avg CPU time: {numpy.mean(scipy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(scipy_wall_times):.6f}s"
        )

        # Test torch_dot on CPU
        torch_cpu_times = []
        torch_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(torch_dot, torch_a, torch_b)
            torch_cpu_times.append(cpu_time)
            torch_wall_times.append(wall_time)

        print(
            f"torch_dot (CPU): Avg CPU time: {numpy.mean(torch_cpu_times):.6f}s, Avg Wall time: {numpy.mean(torch_wall_times):.6f}s"
        )

        # Test torch_dot on GPU
        if torch.cuda.is_available():
            torch_a_gpu = torch_a.cuda()
            torch_b_gpu = torch_b.cuda()
            torch_gpu_times = []
            for _ in range(num_runs):
                _, gpu_time, _ = measure_time_gpu(torch_dot, torch_a_gpu, torch_b_gpu)
                torch_gpu_times.append(gpu_time)

            print(f"torch_dot (GPU): Avg GPU time: {numpy.mean(torch_gpu_times):.6f}s")
        else:
            print("CUDA is not available. Skipping GPU tests.")


# Test with different matrix sizes
matrix_sizes = [
    (100, 100, 100),
    (500, 500, 500),
    (1000, 1000, 1000),
    (2000, 1000, 500),
    (1000, 2000, 1000),
    (1000, 4000, 1000),
]

test_dot_performance(matrix_sizes)
