from lalagrad import Tensor
from time import time
from numpy import array

def lala_matmul(loop):
        x = Tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        s = time()
        for a in range(loop):
                y = Tensor([[a, a, a], [a, a, a], [a, a, a]])
                x.matmul(y)
            
        return time() - s


def np_matmul(loop):
        x = array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        s = time()
        for a in range(loop):
                y = array([[a, a, a], [a, a, a], [a, a, a]])
                x.__matmul__(y)
            
        return time() - s
    
    
def lala_elwise(loop):
        x = Tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        s = time()
        for a in range(loop):
                y = Tensor([[a, a, a], [a, a, a], [a, a, a]])
                x + y            
        return time() - s


def np_elwise(loop):
        x = array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        s = time()
        for a in range(loop):
                y = array([[a, a, a], [a, a, a], [a, a, a]])
                x + y
            
        return time() - s

def lala_mul(loop, axis=0):
        s = time()
        for a in range(loop):
                y = Tensor([[a, a, a], [a, a, a], [a, a, a]])
                y.mul(axis)
            
        return time() - s

def np_mul(loop, axis=0):
        s = time()
        for a in range(loop):
                y = array([[a, a, a], [a, a, a], [a, a, a]])
                y.prod(axis=axis)
            
        return time() - s
    