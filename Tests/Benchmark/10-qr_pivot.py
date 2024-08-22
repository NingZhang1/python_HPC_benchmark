import numpy
import torch
import time
import backend._config

backend._config.disable_fftw()
import backend._numpy
import backend._scipy
import backend._torch

backend._torch.disable_gpu()

numpy_qr = backend._numpy.qr_col_pivoting
scipy_qr = backend._scipy.qr_col_pivoting
torch_qr = backend._torch.qr_col_pivoting


def measure_time(func, *args):
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    result = func(*args)

    end_cpu = time.process_time()
    end_wall = time.perf_counter()

    return result, end_cpu - start_cpu, end_wall - start_wall


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
            _, cpu_time, wall_time = measure_time(numpy_qr, np_matrix, 1e-8, None, "r")
            numpy_cpu_times.append(cpu_time)
            numpy_wall_times.append(wall_time)

        print(
            f"numpy_qr: Avg CPU time: {numpy.mean(numpy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(numpy_wall_times):.6f}s"
        )

        # Test scipy_qr
        scipy_cpu_times = []
        scipy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time(scipy_qr, np_matrix, 1e-8, None, "r")
            scipy_cpu_times.append(cpu_time)
            scipy_wall_times.append(wall_time)

        print(
            f"scipy_qr: Avg CPU time: {numpy.mean(scipy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(scipy_wall_times):.6f}s"
        )

        # Test torch_qr
        torch_cpu_times = []
        torch_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time(torch_qr, torch_matrix, 1e-8, None, "r")
            torch_cpu_times.append(cpu_time)
            torch_wall_times.append(wall_time)

        print(
            f"torch_qr: Avg CPU time: {numpy.mean(torch_cpu_times):.6f}s, Avg Wall time: {numpy.mean(torch_wall_times):.6f}s"
        )


# Test with different matrix sizes
matrix_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 1000), (1000, 2000)]
test_qr_performance(matrix_sizes)
