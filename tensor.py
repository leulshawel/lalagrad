import numpy as np
from typing import Optional, Union, List

from dtype import Dtype
from array_ops import flatten, array_from_shape, add_const,\
    shape_from_array, reverse, num_of_elems
from ops import binary_op_wrapper, unary_op_wrapper

class Tensor():
    __slots__ = "data", "shape", "dtype", "ctx", "strong" #No More attributes
                 
    def __init__(self, data: Optional[Union[None, np.ndarray, List, int, float, bool]]=None, 
        shape: tuple[int]=None, dtype: Optional[Dtype]=None, ctx = None, strong: bool=True):
        assert dtype is None or isinstance(dtype, Dtype), "dtype unknown"
        self.strong = strong
        if data is None:  
            assert shape is not None and dtype is not None, "shape and dtype are required if data is None"
            self.data, self.shape = flatten(array_from_shape(shape)), shape
        else: 
            if isinstance(data, (list, tuple)): self.data, self.shape = flatten(data), reverse(shape_from_array(data))
            else: raise RuntimeError("Tensor creation failed")
        self.dtype = dtype
        
    @classmethod
    def new(cls, data=None, shape=None, dtype=None, ctx=None, strong=None): return cls(data, shape, dtype, ctx, strong)
    #on self or return binary ops
    @binary_op_wrapper
    def __add__(self, other): return [x+y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __sub__(self, other): return [x-y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __mul__(self, other): return [x*y for x, y in zip(self.data, other.data)]
    
    #on self or return unaryOps
    @unary_op_wrapper
    def sadd(self, s): return add_const(self.data, s)
    @unary_op_wrapper
    def __sqr__(self): return self * self
    @unary_op_wrapper
    def __not__(self):  return [not b for b in self.data] if self.dtype==Dtype('bool') else []
    
    #on self ops
    def set(self, val: Union[int, float, bool]): self.data = set(self.data, val)
    #TODO: expand the tensor in a dimension
    def expand(sz, dim): pass
    #reshaping the tensor is simple
    def reshape(self, shape):
        assert num_of_elems(self.shape) == num_of_elems(shape), "Tensor can't be of this shape"
        self.shape = shape
    
    #TODO: reduce Ops  
    #dot product
    def dot(self, other): pass
    #add elemens in a single dimension
    def sum(self, dim): pass
    #multopy two tensors
    def matmul(self, other):
        shape = self.shape
    def mul(self, other): return self * other
    