import numpy
import torch
import time
import BackEnd._config

BackEnd._config.disable_fftw()
# BackEnd._config.enable_torch_gpu_qr_pivoting()

import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_qr = BackEnd._numpy.qr_col_pivoting
scipy_qr = BackEnd._scipy.qr_col_pivoting
torch_qr = BackEnd._torch.qr_col_pivoting


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


def test_qr_performance(sizes, num_runs=5):
    for m, n in sizes:
        print(f"\nTesting performance for matrix size {m}x{n}")

        # Create a random matrix
        np_matrix = numpy.random.rand(m, n)
        torch_matrix = torch.from_numpy(np_matrix)

        # Test numpy_qr
        numpy_cpu_times = []
        numpy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                numpy_qr, np_matrix, 1e-8, None, "r"
            )
            numpy_cpu_times.append(cpu_time)
            numpy_wall_times.append(wall_time)

        print(
            f"numpy_qr: Avg CPU time: {numpy.mean(numpy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(numpy_wall_times):.6f}s"
        )

        # Test scipy_qr
        scipy_cpu_times = []
        scipy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                scipy_qr, np_matrix, 1e-8, None, "r"
            )
            scipy_cpu_times.append(cpu_time)
            scipy_wall_times.append(wall_time)

        print(
            f"scipy_qr: Avg CPU time: {numpy.mean(scipy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(scipy_wall_times):.6f}s"
        )

        # Test torch_qr on CPU
        torch_cpu_times = []
        torch_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                torch_qr, torch_matrix, 1e-8, None, "r"
            )
            torch_cpu_times.append(cpu_time)
            torch_wall_times.append(wall_time)

        print(
            f"torch_qr (CPU): Avg CPU time: {numpy.mean(torch_cpu_times):.6f}s, Avg Wall time: {numpy.mean(torch_wall_times):.6f}s"
        )

        # Test torch_qr on GPU
        if torch.cuda.is_available():
            torch_matrix_gpu = torch_matrix.cuda()
            torch_gpu_times = []
            for _ in range(num_runs):
                _, gpu_time, _ = measure_time_gpu(
                    torch_qr, torch_matrix_gpu, 1e-8, None, "r"
                )
                torch_gpu_times.append(gpu_time)

            print(f"torch_qr (GPU): Avg GPU time: {numpy.mean(torch_gpu_times):.6f}s")
        else:
            print("CUDA is not available. Skipping GPU test.")


# Test with different matrix sizes
matrix_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 1000), (1000, 2000)]
test_qr_performance(matrix_sizes)
