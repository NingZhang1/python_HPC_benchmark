import numpy, scipy
import backend._pyfftw as _pyfftw
from backend._config import ENABLE_FFTW

################# toTensor #################


def toTensor(data):
    return numpy.asarray(data)


def toNumpy(data):
    if not isinstance(data, numpy.ndarray):
        raise ValueError("Data is not a numpy array")
    return data


################# FFT #################


if _pyfftw.FFTW_FOUND and ENABLE_FFTW:

    rfftn = _pyfftw.rfftn
    irfftn = _pyfftw.irfftn
    fftn = _pyfftw.fftn
    ifftn = _pyfftw.ifftn

else:

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.rfftn(x, s, axes, overwrite_x=overwrite_input, workers=threads)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.irfftn(
            x, s, axes, overwrite_x=overwrite_input, workers=threads
        )

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.fftn(x, s, axes, overwrite_x=overwrite_input, workers=threads)

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return scipy.fft.ifftn(x, s, axes)


################# QR #################
