import numpy
import torch
import time
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_cholesky = BackEnd._numpy.cholesky
scipy_cholesky = BackEnd._scipy.cholesky
torch_cholesky = BackEnd._torch.cholesky


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


def create_positive_definite_matrix(n):
    A = numpy.random.rand(n, n)
    return numpy.dot(A, A.T)


def test_cholesky_performance(sizes, num_runs=5):
    for n in sizes:
        print(f"\nTesting performance for matrix size {n}x{n}")

        # Create a random positive definite matrix
        np_matrix = create_positive_definite_matrix(n)
        torch_matrix = torch.from_numpy(np_matrix)

        # Test numpy_cholesky
        numpy_cpu_times = []
        numpy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                numpy_cholesky, np_matrix, True, False, None
            )
            numpy_cpu_times.append(cpu_time)
            numpy_wall_times.append(wall_time)

        print(
            f"numpy_cholesky: Avg CPU time: {numpy.mean(numpy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(numpy_wall_times):.6f}s"
        )

        # Test scipy_cholesky
        scipy_cpu_times = []
        scipy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                scipy_cholesky, np_matrix, True, False, None
            )
            scipy_cpu_times.append(cpu_time)
            scipy_wall_times.append(wall_time)

        print(
            f"scipy_cholesky: Avg CPU time: {numpy.mean(scipy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(scipy_wall_times):.6f}s"
        )

        # Test torch_cholesky on CPU
        torch_cpu_times = []
        torch_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                torch_cholesky, torch_matrix, True, False, None
            )
            torch_cpu_times.append(cpu_time)
            torch_wall_times.append(wall_time)

        print(
            f"torch_cholesky (CPU): Avg CPU time: {numpy.mean(torch_cpu_times):.6f}s, Avg Wall time: {numpy.mean(torch_wall_times):.6f}s"
        )

        # Test torch_cholesky on GPU
        if torch.cuda.is_available():
            torch_matrix_gpu = torch_matrix.cuda()
            torch_gpu_times = []
            for _ in range(num_runs):
                _, gpu_time, _ = measure_time_gpu(
                    torch_cholesky, torch_matrix_gpu, True, False, None
                )
                torch_gpu_times.append(gpu_time)

            print(
                f"torch_cholesky (GPU): Avg GPU time: {numpy.mean(torch_gpu_times):.6f}s"
            )
        else:
            print("CUDA is not available. Skipping GPU test.")


# Test with different matrix sizes
matrix_sizes = [100, 500, 1000, 2000, 5000]
test_cholesky_performance(matrix_sizes)
