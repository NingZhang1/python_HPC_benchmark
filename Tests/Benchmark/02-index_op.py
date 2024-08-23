import numpy as np
import torch
import BackEnd._config
import time

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._torch

numpy_index_add = BackEnd._numpy.index_add
scipy_index_add = BackEnd._scipy.index_add
torch_index_add = BackEnd._torch.index_add

numpy_index_copy = BackEnd._numpy.index_copy
scipy_index_copy = BackEnd._scipy.index_copy
torch_index_copy = BackEnd._torch.index_copy

numpy_take = BackEnd._numpy.take
scipy_take = BackEnd._scipy.take
torch_take = BackEnd._torch.take


def measure_time_cpu(func, *args, **kwargs):
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    result = func(*args, **kwargs)

    end_cpu = time.process_time()
    end_wall = time.perf_counter()

    return result, end_cpu - start_cpu, end_wall - start_wall


def measure_time_gpu(func, *args, **kwargs):
    # Warm-up
    for _ in range(5):
        func(*args, **kwargs)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    result = func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

    return result, elapsed_time, elapsed_time


def performance_test_index_add(A, dim, index, source, alpha=1, num_runs=10):
    print(
        f"\nTesting index_add performance for shape {A.shape}, dim {dim}, index size {len(index)}"
    )

    A_np = A.copy()
    A_sp = A.copy()
    A_torch = torch.from_numpy(A.copy())
    source_torch = torch.from_numpy(source)
    index_torch = torch.from_numpy(index)

    # CPU tests
    for name, func, a, s, i in [
        ("NumPy", numpy_index_add, A_np, source, index),
        ("SciPy", scipy_index_add, A_sp, source, index),
        ("PyTorch (CPU)", torch_index_add, A_torch, source_torch, index_torch),
    ]:
        cpu_times = []
        wall_times = []
        for _ in range(num_runs):
            A_test = a.copy() if isinstance(a, np.ndarray) else a.clone()
            _, cpu_time, wall_time = measure_time_cpu(
                func, A_test, dim, i, s, alpha=alpha
            )
            cpu_times.append(cpu_time)
            wall_times.append(wall_time)
        print(
            f"{name} index_add: Avg CPU time: {np.mean(cpu_times):.6f}s, Avg Wall time: {np.mean(wall_times):.6f}s"
        )

    # GPU test
    if torch.cuda.is_available():
        A_torch_gpu = A_torch.cuda()
        source_torch_gpu = source_torch.cuda()
        index_torch_gpu = index_torch.cuda()
        gpu_times = []
        for _ in range(num_runs):
            A_test = A_torch_gpu.clone()
            _, gpu_time, _ = measure_time_gpu(
                torch_index_add,
                A_test,
                dim,
                index_torch_gpu,
                source_torch_gpu,
                alpha=alpha,
            )
            gpu_times.append(gpu_time)
        print(f"PyTorch (GPU) index_add: Avg GPU time: {np.mean(gpu_times):.6f}s")
    else:
        print("CUDA is not available. Skipping GPU test.")


def performance_test_index_copy(A, dim, index, source, num_runs=10):
    print(
        f"\nTesting index_copy performance for shape {A.shape}, dim {dim}, index size {len(index)}"
    )

    A_np = A.copy()
    A_sp = A.copy()
    A_torch = torch.from_numpy(A.copy())
    source_torch = torch.from_numpy(source)
    index_torch = torch.from_numpy(index)

    # CPU tests
    for name, func, a, s, i in [
        ("NumPy", numpy_index_copy, A_np, source, index),
        ("SciPy", scipy_index_copy, A_sp, source, index),
        ("PyTorch (CPU)", torch_index_copy, A_torch, source_torch, index_torch),
    ]:
        cpu_times = []
        wall_times = []
        for _ in range(num_runs):
            A_test = a.copy() if isinstance(a, np.ndarray) else a.clone()
            _, cpu_time, wall_time = measure_time_cpu(func, A_test, dim, i, s)
            cpu_times.append(cpu_time)
            wall_times.append(wall_time)
        print(
            f"{name} index_copy: Avg CPU time: {np.mean(cpu_times):.6f}s, Avg Wall time: {np.mean(wall_times):.6f}s"
        )

    # GPU test
    if torch.cuda.is_available():
        A_torch_gpu = A_torch.cuda()
        source_torch_gpu = source_torch.cuda()
        index_torch_gpu = index_torch.cuda()
        gpu_times = []
        for _ in range(num_runs):
            A_test = A_torch_gpu.clone()
            _, gpu_time, _ = measure_time_gpu(
                torch_index_copy, A_test, dim, index_torch_gpu, source_torch_gpu
            )
            gpu_times.append(gpu_time)
        print(f"PyTorch (GPU) index_copy: Avg GPU time: {np.mean(gpu_times):.6f}s")
    else:
        print("CUDA is not available. Skipping GPU test.")


def performance_test_take(a, indices, axis=None, num_runs=10):
    print(
        f"\nTesting take performance for shape {a.shape}, indices size {len(indices)}, axis {axis}"
    )

    a_torch = torch.from_numpy(a)
    indices_torch = torch.from_numpy(indices)

    # CPU tests
    for name, func, arr, ind in [
        ("NumPy", numpy_take, a, indices),
        ("SciPy", scipy_take, a, indices),
        ("PyTorch (CPU)", torch_take, a_torch, indices_torch),
    ]:
        cpu_times = []
        wall_times = []
        for _ in range(num_runs):
            _, cpu_time, wall_time = measure_time_cpu(func, arr, ind, axis=axis)
            cpu_times.append(cpu_time)
            wall_times.append(wall_time)
        print(
            f"{name} take: Avg CPU time: {np.mean(cpu_times):.6f}s, Avg Wall time: {np.mean(wall_times):.6f}s"
        )

    # GPU test
    if torch.cuda.is_available():
        a_torch_gpu = a_torch.cuda()
        indices_torch_gpu = indices_torch.cuda()
        gpu_times = []
        for _ in range(num_runs):
            _, gpu_time, _ = measure_time_gpu(
                torch_take, a_torch_gpu, indices_torch_gpu, axis=axis
            )
            gpu_times.append(gpu_time)
        print(f"PyTorch (GPU) take: Avg GPU time: {np.mean(gpu_times):.6f}s")
    else:
        print("CUDA is not available. Skipping GPU test.")


def run_performance_tests():
    # Test case 1: 2D array, add to rows
    A = np.random.rand(10000, 1000)
    index = np.random.choice(10000, 1000, replace=False)
    source = np.random.rand(1000, 1000)
    performance_test_index_add(A, 0, index, source)
    performance_test_index_copy(A, 0, index, source)

    # Test case 2: 2D array, add to columns
    A = np.random.rand(1000, 10000)
    index = np.random.choice(10000, 1000, replace=False)
    source = np.random.rand(1000, 1000)
    performance_test_index_add(A, 1, index, source)
    performance_test_index_copy(A, 1, index, source)

    # Test case 3: 3D array
    A = np.random.rand(100, 100, 100)
    index = np.random.choice(100, 10, replace=False)
    source = np.random.rand(10, 100, 100)
    performance_test_index_add(A, 0, index, source)
    performance_test_index_copy(A, 0, index, source)

    # Test case 4: take function
    a = np.random.rand(10000, 1000)
    indices = np.random.choice(10000, 1000, replace=False)
    performance_test_take(a, indices, axis=0)
    indices2 = np.random.choice(1000, 250, replace=False)
    performance_test_take(a, indices2, axis=1)
    # performance_test_take(a.ravel(), indices)


if __name__ == "__main__":
    run_performance_tests()
