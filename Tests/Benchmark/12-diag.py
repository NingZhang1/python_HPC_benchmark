import numpy
import torch
import time
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_eigh = BackEnd._numpy.eigh
scipy_eigh = BackEnd._scipy.eigh
torch_eigh = BackEnd._torch.eigh


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


def create_hermitian_matrix(n):
    A = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
    return A + A.conj().T


def test_eigh_performance(sizes, num_runs=5):
    for n in sizes:
        print(f"\nTesting performance for matrix size {n}x{n}")

        # Create a random Hermitian matrix
        A = create_hermitian_matrix(n)

        # Convert to PyTorch tensor
        A_torch = torch.from_numpy(A)

        # Test numpy_eigh
        numpy_cpu_times = []
        numpy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(numpy_eigh, A)
            numpy_cpu_times.append(cpu_time)
            numpy_wall_times.append(wall_time)

        print(
            f"numpy_eigh: Avg CPU time: {numpy.mean(numpy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(numpy_wall_times):.6f}s"
        )

        # Test scipy_eigh
        scipy_cpu_times = []
        scipy_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(scipy_eigh, A)
            scipy_cpu_times.append(cpu_time)
            scipy_wall_times.append(wall_time)

        print(
            f"scipy_eigh: Avg CPU time: {numpy.mean(scipy_cpu_times):.6f}s, Avg Wall time: {numpy.mean(scipy_wall_times):.6f}s"
        )

        # Test torch_eigh on CPU
        torch_cpu_times = []
        torch_wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(torch_eigh, A_torch)
            torch_cpu_times.append(cpu_time)
            torch_wall_times.append(wall_time)

        print(
            f"torch_eigh (CPU): Avg CPU time: {numpy.mean(torch_cpu_times):.6f}s, Avg Wall time: {numpy.mean(torch_wall_times):.6f}s"
        )

        # Test torch_eigh on GPU
        if torch.cuda.is_available():
            A_torch_gpu = A_torch.cuda()
            torch_gpu_times = []
            for _ in range(num_runs):
                _, gpu_time, _ = measure_time_gpu(torch_eigh, A_torch_gpu)
                torch_gpu_times.append(gpu_time)

            print(f"torch_eigh (GPU): Avg GPU time: {numpy.mean(torch_gpu_times):.6f}s")
        else:
            print("CUDA is not available. Skipping GPU test.")


# Test with different matrix sizes
matrix_sizes = [100, 500, 1000, 2000]
test_eigh_performance(matrix_sizes)
