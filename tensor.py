import numpy as np
from dtype import Dtype
from typing import Optional, Union, List
from array_ops import flatten, array_from_shape, add_const

class Tensor():
    __slots__ = "data", "shape", "dtype", "ctx", "ret" #No More attributes
    
    def __init__(self, data: Optional[Union[None, np.ndarray, List, int, float, bool]]=None, shape: tuple[int]=None, dtype: Optional[Dtype]=None, ctx = None, ret: bool=True):
        assert dtype is None or isinstance(dtype, Dtype), "dtype unknown"
        self.ret = ret
        if data is None:  
            assert shape is not None and dtype is not None, "shape and dtype are required if data is None"
            self.data, self.shape, self.dtype = flatten(array_from_shape(shape)), shape, dtype  
        else: 
            if isinstance(data, (list, tuple)): self.data, self.shape, self.dtype = flatten(data), None, None 
            else: raise RuntimeError("Tensor creation failed")
        
    #How do you want the result of your ops on this Tensor
    def binary_op_ret_handler(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.ret and other.ret: return Tensor(data=value)
            if self.ret: other.data = value
            else: self.data = value
        return wrapper
    
    #How do you want the result of your ops on this Tensor
    def unary_op_ret_handler(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.ret: return Tensor(data=value)
            self.data = value
        return wrapper
            
    @binary_op_ret_handler
    def __add__(self, other):
        assert self.shape==other.shape, "Incompatible dtype or shape"
        assert self.dtype==other.dtype, f"can't ADD dtype {self.dtype} and dtype {other.dtype}"
        return self.data + other.data
    @binary_op_ret_handler
    def __sub__(self, other): 
        assert self.shape==other.shape, "Incompatible dtype or shape"
        assert self.dtype==other.dtype, f"can't SUB dtype {self.dtype} and dtype {other.dtype}"
        return self.data - other.data 
    @binary_op_ret_handler
    def __mul__(self, other): 
        assert self.shape==other.shape, "Incompatible dtype or shape"
        assert self.dtype==other.dtype, f"can't MUL dtype {self.dtype} and dtype {other.dtype}"
        return self.data * other.data 
    def __repr__(self): return f"<Tensor of Shape: {self.shape} dtype: {self.dtype}>"
    @unary_op_ret_handler
    def sadd(self, s): return add_const(self.data, s)
    @unary_op_ret_handler
    def square(self): return self * self
    @unary_op_ret_handler
    def neg(slef):  pass
    def set(self, val: Union[int, float, bool]): self.data = set(self.data, val)
    
    