from BackEnd._num_threads import num_threads

try:
    import pyfftw

    FFTW_FOUND = True
    # implement FFTW obj #
except ImportError:
    FFTW_FOUND = False
    # implement FFTW obj #
