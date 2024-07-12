"""
https://numba.readthedocs.io/en/stable/reference/pysupported.html#random
https://github.com/numba/numba/pull/7068
"""
import numba as nb
import numpy as np
from numba import njit
import random
from pathlib import Path

@njit
def seed(a):
    random.seed(a)

@njit
def rand():
    return random.random()

@nb.njit(
    (nb.float64[:],),
    parallel=True,
    nogil=True,
    cache=True,
)
def fill_me(arr):
    for i in nb.prange(len(arr)):
        arr[i] = random.normalvariate(mu=0, sigma=1)
    return

if __name__ == "__main__":
    # seed(0)
    # for i in range(10):
    #     print(rand())
    seed(0)
    arr = np.zeros(100_000, dtype=np.float64)
    fill_me(arr)
    
    if not Path("random_numbers.npy").exists():
        np.save("random_numbers.npy", arr)
    else:
        ref_arr = np.load("random_numbers.npy")
        assert np.allclose(arr, ref_arr)

    
