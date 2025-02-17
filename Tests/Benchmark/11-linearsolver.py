import numpy
import torch
import time
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_solve_cholesky = BackEnd._numpy.solve_cholesky
scipy_solve_cholesky = BackEnd._scipy.solve_cholesky
torch_solve_cholesky = BackEnd._torch.solve_cholesky


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


def test_solve_cholesky_performance(sizes, num_runs=5):
    for n, m in sizes:
        print(f"\nTesting performance for A: {n}x{n}, B: {n}x{m}")

        # Create a random positive definite matrix A and a random matrix B
        A = create_positive_definite_matrix(n)
        B = numpy.random.rand(n, m)

        # Convert to PyTorch tensors
        A_torch = torch.from_numpy(A)
        B_torch = torch.from_numpy(B)

        # Test numpy_solve_cholesky
        numpy_cpu_times = []
        numpy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                numpy_solve_cholesky, A.copy(), B.copy()
            )
            numpy_cpu_times.append(cpu_time)
            numpy_wall_times.append(wall_time)

        print(
            f"numpy_solve_cholesky: Avg CPU time: {numpy.mean(numpy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(numpy_wall_times):.6f}s"
        )

        # Test scipy_solve_cholesky
        scipy_cpu_times = []
        scipy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                scipy_solve_cholesky, A.copy(), B.copy()
            )
            scipy_cpu_times.append(cpu_time)
            scipy_wall_times.append(wall_time)

        print(
            f"scipy_solve_cholesky: Avg CPU time: {numpy.mean(scipy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(scipy_wall_times):.6f}s"
        )

        # Test torch_solve_cholesky on CPU
        torch_cpu_times = []
        torch_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(
                torch_solve_cholesky, A_torch.clone(), B_torch.clone()
            )
            torch_cpu_times.append(cpu_time)
            torch_wall_times.append(wall_time)

        print(
            f"torch_solve_cholesky (CPU): Avg CPU time: {numpy.mean(torch_cpu_times):.6f}s, Avg Wall time: {numpy.mean(torch_wall_times):.6f}s"
        )

        # Test torch_solve_cholesky on GPU
        if torch.cuda.is_available():
            A_torch_gpu = A_torch.cuda()
            B_torch_gpu = B_torch.cuda()
            torch_gpu_times = []
            for _ in range(num_runs):
                _, gpu_time, _ = measure_time_gpu(
                    torch_solve_cholesky, A_torch_gpu.clone(), B_torch_gpu.clone()
                )
                torch_gpu_times.append(gpu_time)

            print(
                f"torch_solve_cholesky (GPU): Avg GPU time: {numpy.mean(torch_gpu_times):.6f}s"
            )
        else:
            print("CUDA is not available. Skipping GPU test.")


# Test with different matrix sizes
matrix_sizes = [(100, 10), (500, 50), (1000, 100), (2000, 200), (5000, 500)]
test_solve_cholesky_performance(matrix_sizes)
