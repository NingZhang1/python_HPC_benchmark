import numpy
import time
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._scipy
import BackEnd._numpy
import BackEnd._pyfftw
import BackEnd._torch

NBUNCH = 96
SIZE = [11, 16, 24, 30, 36, 48]
NUMBER = 10  # Number of FFT executions per test
REPEAT = 10  # Number of times to repeat the entire test


def run_test_cpu(func, a, size):
    start_cpu = time.process_time()
    start_wall = time.time()

    for _ in range(NUMBER):
        func(a, s=(size, size, size), axes=(1, 2, 3), overwrite_input=True)

    end_cpu = time.process_time()
    end_wall = time.time()

    return end_cpu - start_cpu, end_wall - start_wall


def run_test_gpu(func, a, size):
    # Warm-up
    for _ in range(5):
        func(a, s=(size, size, size), axes=(1, 2, 3), overwrite_input=True)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(NUMBER):
        func(a, s=(size, size, size), axes=(1, 2, 3), overwrite_input=True)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

    return elapsed_time, elapsed_time


def run_tests_cpu(size):
    a = numpy.random.rand(NBUNCH, size, size, size)

    tests = [
        ("NumPy FFT", BackEnd._numpy.fftn),
        ("SciPy FFT", BackEnd._scipy.fftn),
        ("pyFFTW", BackEnd._pyfftw.fftn),
    ]

    results = {}
    for name, func in tests:
        cpu_times = []
        wall_times = []
        for _ in range(REPEAT):
            cpu_time, wall_time = run_test_cpu(func, a, size)
            cpu_times.append(cpu_time)
            wall_times.append(wall_time)

        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        avg_wall_time = sum(wall_times) / len(wall_times)
        results[name] = (avg_cpu_time, avg_wall_time)

    return results


def run_tests_gpu(size):
    a = numpy.random.rand(NBUNCH, size, size, size)
    a = BackEnd._torch.toTensor(a).cuda()

    tests = [
        ("torch FFT (GPU)", BackEnd._torch.fftn),
    ]

    results = {}
    for name, func in tests:
        gpu_times = []
        for _ in range(REPEAT):
            gpu_time, _ = run_test_gpu(func, a, size)
            gpu_times.append(gpu_time)

        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        results[name] = (avg_gpu_time, avg_gpu_time)

    return results


print(
    f"Test configuration: {REPEAT} repeats for each size, {NUMBER} FFT executions per test"
)

for size in SIZE:
    print(f"\nTesting size = {size}")

    # CPU tests
    results_cpu = run_tests_cpu(size)
    for name, (cpu_time, wall_time) in results_cpu.items():
        print(f"  {name}:")
        print(f"    CPU time: {cpu_time:.6f} seconds")
        print(f"    Wall time: {wall_time:.6f} seconds")

    # GPU tests
    if torch.cuda.is_available():
        results_gpu = run_tests_gpu(size)
        for name, (gpu_time, _) in results_gpu.items():
            print(f"  {name}:")
            print(f"    GPU time: {gpu_time:.6f} seconds")
    else:
        print("CUDA is not available. Skipping GPU tests.")
