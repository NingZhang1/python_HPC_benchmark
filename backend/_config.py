ENABLE_FFTW = True
ENABLE_PYSCF_LIB = True
FORCE_PYSCF_LIB = False


def enable_fftw():
    global ENABLE_FFTW
    ENABLE_FFTW = True


def disable_fftw():
    global ENABLE_FFTW
    ENABLE_FFTW = False


def enable_pyscf_lib():
    global ENABLE_PYSCF_LIB
    ENABLE_PYSCF_LIB = True


def disable_pyscf_lib():
    global ENABLE_PYSCF_LIB
    ENABLE_PYSCF_LIB = False


def force_pyscf_lib():
    global FORCE_PYSCF_LIB
    FORCE_PYSCF_LIB = True


def disable_pyscf_lib_force():
    global FORCE_PYSCF_LIB
    FORCE_PYSCF_LIB = False
