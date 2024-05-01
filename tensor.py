import numpy as np
from typing import Optional, Union, List

from lalagrad.  dtype import Dtype
from lalagrad.array_ops import flatten, array_from_shape, add_const,\
    shape_from_array, reverse, num_of_elems, scale, devid_list
    
from lalagrad.ops import binary_op_wrapper, unary_op_wrapper

class Tensor():
    __slots__ = "data", "shape", "dtype", "ctx", "strong", "mat" 
    class Matrix:
        #TODO: implement matrix multiplicationhttps://github.com/karpathy/micrograd
        def matmul(self, other):
            assert isinstance(other, Tensor.Matrix),\
                "matrix multiplication is only defined for 2D Tensors"
            pass
    
    def __init__(self, data: Optional[Union[None, List, int, float, bool]]=None, 
        shape: tuple[int]=None, dtype: Optional[Dtype]=None, ctx = None, strong: bool=True):
        assert dtype is None or isinstance(dtype, Dtype), "dtype unknown"
        self.strong = strong
        #if not isinstance(data[0], (list, tuple)): data = [data] #No vector so far
        if data is None:  
            assert shape is not None and dtype is not None, "shape and dtype are required if data is None"
            self.data, self.shape = flatten(array_from_shape(shape)), shape
        else: 
            if isinstance(data, (list, tuple)): self.data, self.shape = flatten(data), tuple(reverse(shape_from_array(data)))
            else: raise RuntimeError("Tensor creation failed")
        self.dtype = dtype
        self.mat = Tensor.Matrix() if len(self.shape) == 2 else None
        
    @classmethod
    def new(cls, data=None, shape=None, dtype=None, ctx=None, strong=None): return cls(data, shape, dtype, ctx, strong)
    #get the data with right dimension (unflatten)
    def data_with_dim(self, l=None, n=0): 
        if not l: l = self.data
        if n+1 == len(self.shape): return l
        return [self.data_with_dim(dl, n+1) for dl in devid_list(l, self.shape[n])]  
            
    #on self or return binary ops
    @binary_op_wrapper  
    def __add__(self, other): return [x+y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __sub__(self, other): return [x-y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __mul__(self, other): return [x*y for x, y in zip(self.data, other.data)]
    def matmul(self, other):  return Tensor(data=self.mat.matmul(other.mat)) 
    
    #on self or return unaryOps
    @unary_op_wrapper
    def sadd(self, s): return add_const(self.data, s)
    #on self or return unaryOps
    @unary_op_wrapper
    def smul(self, s): return scale(self.data, s)
    @unary_op_wrapper
    def __sqr__(self): return self * self
    @unary_op_wrapper
    def __not__(self):  return [not b for b in self.data] if self.dtype==Dtype('bool') else []
    
    #on self ops
    def set(self, val: Union[int, float, bool]): self.data = set(self.data, val)
    #TODO: expand the tensor in a axis
    def expand(self, shape, n=0): pass
                    
    def transpose(self, axis1, axis2): self.shape[axis1], self.shape2 = self.shape[axis2], self.shape[axis1]
    #reshaping the tensor is simple
    def reshape(self, shape):
        assert num_of_elems(self.shape) == num_of_elems(shape), "Tensor can't be of this shape"
        self.mat = Tensor.Matrix() if len(shape) == 2 else None
        self.shape = tuple(shape)
    
    #TODO: reduce Ops  
    #dot product
    def dot(self, other, axis): pass
    #add elemens in a single axis
    def sum(self, axis): pass
    #elemnt-wise multiplication
    def mul(self, other): return self * other