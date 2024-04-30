import numpy as np
from typing import Optional, Union, List

from dtype import Dtype
from array_ops import flatten, array_from_shape, add_const, shape_from_array, reverse
from ops import binary_op_ret_handler, unary_op_ret_handler

class Tensor():
    __slots__ = "data", "shape", "dtype", "ctx", "strong" #No More attributes
    
    def __init__(self, data: Optional[Union[None, np.ndarray, List, int, float, bool]]=None, shape: tuple[int]=None, dtype: Optional[Dtype]=None, ctx = None, strong: bool=True):
        assert dtype is None or isinstance(dtype, Dtype), "dtype unknown"
        self.strong = strong
        if data is None:  
            assert shape is not None and dtype is not None, "shape and dtype are required if data is None"
            self.data, self.shape, self.dtype = flatten(array_from_shape(shape)), shape, dtype  
        else: 
            if isinstance(data, (list, tuple)): self.data, self.shape, self.dtype = flatten(data), reverse(shape_from_array(data)), None 
            else: raise RuntimeError("Tensor creation failed")
    
    @classmethod
    def new(cls, data=None, shape=None, dtype=None, ctx=None, strong=None):return cls(data, shape, dtype, ctx, strong)
    @binary_op_ret_handler
    def __add__(self, other):
        assert self.shape==other.shape and self.dtype.dtype==other.dtype.dtype, f"can't ADD {self} and {other}"
        return self.data + other.data
    @binary_op_ret_handler
    def __sub__(self, other): 
        assert self.shape==other.shape and self.dtype==other.dtype, f"can't SUB {self} and {other}"
        return self.data - other.data 
    @binary_op_ret_handler
    def __mul__(self, other): 
        assert self.shape==other.shape and self.dtype==other.dtype, f"can't MUL {self} and {other}"
        return self.data * other.data 
    def __repr__(self): return f"<Tensor of Shape: {self.shape} dtype: {self.dtype} strength: {self.strong}>"
    @unary_op_ret_handler
    def sadd(self, s): return add_const(self.data, s)
    @unary_op_ret_handler
    def square(self): return self * self
    @unary_op_ret_handler
    def neg(slef):  pass
    def set(self, val: Union[int, float, bool]): self.data = set(self.data, val)
    
    