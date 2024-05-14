from lalagrad import Tensor
from time import time
from numpy import array


def lala_matmul(times=1000000):
    x = Tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
    s = time()
    for a in range(times):
        y = Tensor([[a, a, a], [a, a, a], [a, a, a]])
        x.matmul(y)
        
    return time() - s


def np_matmul(times=1000000):
    x = array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
    s = time()
    for a in range(times):
        y = array([[a, a, a], [a, a, a], [a, a, a]])
        x.__matmul__(y)
        
    return time() - s

lala = lala_matmul()
np = np_matmul()
ratio = lala / np
print(f"lala: {lala}")
print(f"numpy: {np}")
print(f"about {ratio} times {'slower' if ratio > 1 else 'faster'}")