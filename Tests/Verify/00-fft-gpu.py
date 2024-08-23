import numpy
import torch
import BackEnd._torch

m, n, k = 13, 17, 19

a = numpy.random.rand(m, n, k)
a_torch = a.copy()
a_torch = BackEnd._torch.toTensor(a_torch)

b1 = BackEnd._torch.fftn(a_torch, s=(m, n, k), axes=(0, 1, 2))
assert isinstance(b1, torch.Tensor)
b1 = BackEnd._torch.toNumpy(b1)

# GPU Test
if torch.cuda.is_available():
    # Move data to GPU
    a_gpu = torch.tensor(a, device="cuda")

    # Perform FFT on GPU
    b2 = BackEnd._torch.fftn(a_gpu, s=(m, n, k), axes=(0, 1, 2))
    assert isinstance(b2, torch.Tensor)
    assert b2.device.type == "cuda"

    # Move result back to CPU for comparison
    b2_cpu = BackEnd._torch.toNumpy(b2.cpu())

    # Compare CPU and GPU results
    assert numpy.allclose(b1, b2_cpu, atol=1e-6), "CPU and GPU results do not match"

    print("CPU and GPU tests passed successfully!")
else:
    print("CUDA is not available. GPU test skipped.")
