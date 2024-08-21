import torch, numpy, scipy
import backend._pyfftw as _pyfftw
from backend._config import ENABLE_FFTW

USE_GPU = True
FFT_CPU_USE_TORCH_ANYWAY = False


def disable_gpu():
    global USE_GPU
    USE_GPU = False


def enable_gpu():
    global USE_GPU
    USE_GPU = True


torch.set_grad_enabled(False)

################# toTensor #################


def toTensor(data, cpu=False):
    if USE_GPU and not cpu:
        return torch.tensor(data, device="cuda")
    else:
        return torch.tensor(data, device="cpu")


def toNumpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    assert isinstance(data, numpy.ndarray)
    return data


################# FFT on cpu/gpu #################

if _pyfftw.FFTW_FOUND and ENABLE_FFTW and not FFT_CPU_USE_TORCH_ANYWAY:

    def _rfftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.rfftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)

    def _irfftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.irfftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)

    def _fftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.fftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)
    
    def _ifftn_cpu(x, s=None, axes=None, overwrite_input=None, threads=None):
        x = toNumpy(x)
        res = _pyfftw.ifftn(
            x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads
        )
        return toTensor(res, cpu=True)

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.rfft(x, s=s, dim=axes)
        else:
            return _rfftn_cpu(x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.irfft(x, s=s, dim=axes)
        else:
            return _irfftn_cpu(x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads)
    
    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.fft.fftn(x, s=s, dim=axes)
        else:
            return _fftn_cpu(x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads)
    
    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        if x.is_cuda:
            return torch.fft.ifftn(x, s=s, dim=axes)
        else:
            return _ifftn_cpu(x, s=s, axes=axes, overwrite_input=overwrite_input, threads=threads)

else:

    print("Using torch for FFT")

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.rfft(x, s=s, dim=axes)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.irfft(x, s=s, dim=axes)

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.fft.fftn(x, s=s, dim=axes)

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return torch.fft.ifftn(x, s=s, dim=axes)
