import numpy as np
import torch
import timeit


def benchmark_numpy(size, rounds=1000):
    arr = np.random.rand(size)

    def fun1():
        arr[:] = 0

    def fun2():
        arr.fill(0)

    slice_time = timeit.timeit(fun1, number=rounds)
    fill_time = timeit.timeit(fun2, number=rounds)

    print(f"NumPy Array Size: {size}")
    print(f"Slice Assignment: {slice_time:.6f} seconds")
    print(f"In-place Fill:    {fill_time:.6f} seconds")
    print(f"Ratio (Slice/Fill): {slice_time/fill_time:.2f}")
    print()


def benchmark_torch(size, rounds=1000):
    tensor = torch.rand(size)

    def fun1():
        tensor[:] = 0

    def fun2():
        tensor.zero_()

    slice_time = timeit.timeit(fun1, number=rounds)
    zero_time = timeit.timeit(fun2, number=rounds)

    print(f"PyTorch Tensor Size: {size}")
    print(f"Slice Assignment: {slice_time:.6f} seconds")
    print(f"In-place Zero:    {zero_time:.6f} seconds")
    print(f"Ratio (Slice/Zero): {slice_time/zero_time:.2f}")
    print()


# Run benchmarks
for size in [100, 10000, 1000000, 100000000]:
    benchmark_numpy(size)
    benchmark_torch(size)
