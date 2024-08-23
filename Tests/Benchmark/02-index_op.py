import numpy as np
import torch
import backend._config

backend._config.disable_fftw()
import backend._numpy
import backend._scipy
import backend._torch

backend._torch.disable_gpu()
import time

numpy_index_add = backend._numpy.index_add
scipy_index_add = backend._scipy.index_add
torch_index_add = backend._torch.index_add

numpy_index_copy = backend._numpy.index_copy
scipy_index_copy = backend._scipy.index_copy
torch_index_copy = backend._torch.index_copy

numpy_take = backend._numpy.take
scipy_take = backend._scipy.take
torch_take = backend._torch.take


def measure_time(func, *args, **kwargs):
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    result = func(*args, **kwargs)

    end_cpu = time.process_time()
    end_wall = time.perf_counter()

    return result, end_cpu - start_cpu, end_wall - start_wall


def performance_test_index_add(A, dim, index, source, alpha=1, num_runs=10):
    print(
        f"\nTesting index_add performance for shape {A.shape}, dim {dim}, index size {len(index)}"
    )

    A_np = A.copy()
    A_sp = A.copy()
    A_torch = torch.from_numpy(A.copy())
    source_torch = torch.from_numpy(source)
    index_torch = torch.from_numpy(index)

    # NumPy performance
    np_cpu_times = []
    np_wall_times = []
    for _ in range(num_runs):
        A_test = A_np.copy()
        _, cpu_time, wall_time = measure_time(
            numpy_index_add, A_test, dim, index, source, alpha=alpha
        )
        np_cpu_times.append(cpu_time)
        np_wall_times.append(wall_time)

    print(
        f"NumPy index_add: Avg CPU time: {np.mean(np_cpu_times):.6f}s, Avg Wall time: {np.mean(np_wall_times):.6f}s"
    )

    # SciPy performance
    sp_cpu_times = []
    sp_wall_times = []
    for _ in range(num_runs):
        A_test = A_sp.copy()
        _, cpu_time, wall_time = measure_time(
            scipy_index_add, A_test, dim, index, source, alpha=alpha
        )
        sp_cpu_times.append(cpu_time)
        sp_wall_times.append(wall_time)

    print(
        f"SciPy index_add: Avg CPU time: {np.mean(sp_cpu_times):.6f}s, Avg Wall time: {np.mean(sp_wall_times):.6f}s"
    )

    # PyTorch performance
    torch_cpu_times = []
    torch_wall_times = []
    for _ in range(num_runs):
        A_test = A_torch.clone()
        _, cpu_time, wall_time = measure_time(
            torch_index_add, A_test, dim, index_torch, source_torch, alpha=alpha
        )
        torch_cpu_times.append(cpu_time)
        torch_wall_times.append(wall_time)

    print(
        f"PyTorch index_add: Avg CPU time: {np.mean(torch_cpu_times):.6f}s, Avg Wall time: {np.mean(torch_wall_times):.6f}s"
    )


def performance_test_index_copy(A, dim, index, source, num_runs=10):
    print(
        f"\nTesting index_copy performance for shape {A.shape}, dim {dim}, index size {len(index)}"
    )

    A_np = A.copy()
    A_sp = A.copy()
    A_torch = torch.from_numpy(A.copy())
    source_torch = torch.from_numpy(source)
    index_torch = torch.from_numpy(index)

    # NumPy performance
    np_cpu_times = []
    np_wall_times = []
    for _ in range(num_runs):
        A_test = A_np.copy()
        _, cpu_time, wall_time = measure_time(
            numpy_index_copy, A_test, dim, index, source
        )
        np_cpu_times.append(cpu_time)
        np_wall_times.append(wall_time)

    print(
        f"NumPy index_copy: Avg CPU time: {np.mean(np_cpu_times):.6f}s, Avg Wall time: {np.mean(np_wall_times):.6f}s"
    )

    # SciPy performance
    sp_cpu_times = []
    sp_wall_times = []
    for _ in range(num_runs):
        A_test = A_sp.copy()
        _, cpu_time, wall_time = measure_time(
            scipy_index_copy, A_test, dim, index, source
        )
        sp_cpu_times.append(cpu_time)
        sp_wall_times.append(wall_time)

    print(
        f"SciPy index_copy: Avg CPU time: {np.mean(sp_cpu_times):.6f}s, Avg Wall time: {np.mean(sp_wall_times):.6f}s"
    )

    # PyTorch performance
    torch_cpu_times = []
    torch_wall_times = []
    for _ in range(num_runs):
        A_test = A_torch.clone()
        _, cpu_time, wall_time = measure_time(
            torch_index_copy, A_test, dim, index_torch, source_torch
        )
        torch_cpu_times.append(cpu_time)
        torch_wall_times.append(wall_time)

    print(
        f"PyTorch index_copy: Avg CPU time: {np.mean(torch_cpu_times):.6f}s, Avg Wall time: {np.mean(torch_wall_times):.6f}s"
    )


def performance_test_take(a, indices, axis=None, num_runs=10):
    print(
        f"\nTesting take performance for shape {a.shape}, indices size {len(indices)}, axis {axis}"
    )

    a_torch = torch.from_numpy(a)
    indices_torch = torch.from_numpy(indices)

    # NumPy performance
    np_cpu_times = []
    np_wall_times = []
    for _ in range(num_runs):
        _, cpu_time, wall_time = measure_time(numpy_take, a, indices, axis=axis)
        np_cpu_times.append(cpu_time)
        np_wall_times.append(wall_time)

    print(
        f"NumPy take: Avg CPU time: {np.mean(np_cpu_times):.6f}s, Avg Wall time: {np.mean(np_wall_times):.6f}s"
    )

    # SciPy performance
    sp_cpu_times = []
    sp_wall_times = []
    for _ in range(num_runs):
        _, cpu_time, wall_time = measure_time(scipy_take, a, indices, axis=axis)
        sp_cpu_times.append(cpu_time)
        sp_wall_times.append(wall_time)

    print(
        f"SciPy take: Avg CPU time: {np.mean(sp_cpu_times):.6f}s, Avg Wall time: {np.mean(sp_wall_times):.6f}s"
    )

    # PyTorch performance
    torch_cpu_times = []
    torch_wall_times = []
    for _ in range(num_runs):
        _, cpu_time, wall_time = measure_time(
            torch_take, a_torch, indices_torch, axis=axis
        )
        torch_cpu_times.append(cpu_time)
        torch_wall_times.append(wall_time)

    print(
        f"PyTorch take: Avg CPU time: {np.mean(torch_cpu_times):.6f}s, Avg Wall time: {np.mean(torch_wall_times):.6f}s"
    )


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
