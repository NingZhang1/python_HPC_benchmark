import numpy
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
    print("Using pyfftw")

    rfftn = _pyfftw.rfftn
    irfftn = _pyfftw.irfftn
    fftn = _pyfftw.fftn
    ifftn = _pyfftw.ifftn

else:
    print("Using numpy.fft")

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.rfftn(x, s, axes)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.irfftn(x, s, axes)

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.fftn(x, s, axes)

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None):
        return numpy.fft.ifftn(x, s, axes)


################# QR #################
