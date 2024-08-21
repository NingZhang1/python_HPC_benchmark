import os, multiprocessing

MULTI_THREADING = True


def num_threads():
    if not MULTI_THREADING:
        return 1
    if "OMP_NUM_THREADS" in os.environ:
        return int(os.environ["OMP_NUM_THREADS"])
    else:
        return multiprocessing.cpu_count()
