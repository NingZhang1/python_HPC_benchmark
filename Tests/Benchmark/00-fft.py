import numpy
import time
import backend._config

backend._config.disable_fftw()
import backend._numpy
import backend._scipy
import backend._pyfftw
import backend._torch

backend._torch.disable_gpu()

NBUNCH = 96
SIZE = [11, 16, 24, 30, 36, 48]
NUMBER = 10  # 每个测试运行的次数
REPEAT = 10  # 重复整个测试的次数


def run_test(func, a, size):
    start_cpu = time.process_time()
    start_wall = time.time()

    for _ in range(NUMBER):
        func(a, s=(size, size, size), axes=(1, 2, 3), overwrite_input=True)

    end_cpu = time.process_time()
    end_wall = time.time()

    return end_cpu - start_cpu, end_wall - start_wall


def run_tests(size):
    a = numpy.random.rand(NBUNCH, size, size, size)

    tests = [
        ("NumPy FFT", backend._numpy.fftn),
        ("SciPy FFT", backend._scipy.fftn),
        ("pyFFTW", backend._pyfftw.fftn),
    ]

    results = {}
    for name, func in tests:
        cpu_times = []
        wall_times = []
        for _ in range(REPEAT):
            cpu_time, wall_time = run_test(func, a, size)
            cpu_times.append(cpu_time)
            wall_times.append(wall_time)

        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        avg_wall_time = sum(wall_times) / len(wall_times)
        results[name] = (avg_cpu_time, avg_wall_time)

    return results


def run_tests_torch(size):
    a = numpy.random.rand(NBUNCH, size, size, size)
    a = backend._torch.toTensor(a)

    tests = [
        ("torch FFT", backend._numpy.fftn),
        # ("SciPy FFT", backend._scipy.fftn),
        # ("pyFFTW", backend._pyfftw.fftn),
    ]

    results = {}
    for name, func in tests:
        cpu_times = []
        wall_times = []
        for _ in range(REPEAT):
            cpu_time, wall_time = run_test(func, a, size)
            cpu_times.append(cpu_time)
            wall_times.append(wall_time)

        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        avg_wall_time = sum(wall_times) / len(wall_times)
        results[name] = (avg_cpu_time, avg_wall_time)

    return results


print(f"测试配置: 每个 size 重复 {REPEAT} 次，每次执行 {NUMBER} 次 FFT")
for size in SIZE:
    print(f"\n测试 size = {size}")
    results = run_tests(size)
    for name, (cpu_time, wall_time) in results.items():
        print(f"  {name}:")
        print(f"    CPU 时间: {cpu_time:.6f} 秒")
        print(f"    墙钟时间: {wall_time:.6f} 秒")
    results = run_tests_torch(size)
    for name, (cpu_time, wall_time) in results.items():
        print(f"  {name}:")
        print(f"    CPU 时间: {cpu_time:.6f} 秒")
        print(f"    墙钟时间: {wall_time:.6f} 秒")
