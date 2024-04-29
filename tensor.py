import numpy as np
from dtype import Dtype
from typing import Optional, Union, List
from helper import flatten, get_shape, Scalar, dtype

class Tensor():
    __slots__ = "shape", "dtype", "ctx", "ret" #No More attributes
    
    def __init__(self, data: Optional[Union[None, np.ndarray, List, Scalar]]=None, shape: tuple[int]=None, dtype: Optional[Dtype]=None, ctx = None, ret: bool=True):
        assert dtype is None or isinstance(dtype, Dtype), "dtype unknown"
        
        if data is None: 
            assert shape is not None and dtype is not None, "shape and dtype are required if data is not provided"
            np.ndarray(shape, dtype)
        else:
            if isinstance(data, (list, tuple)): self.data, self.shape = flatten(data), None
            else: raise RuntimeError("Tensor creatin failed")
        
    #How do you want the result of your ops on this Tensor
    def ret_handler(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.ret and other.ret: return value
            if self.ret: other.data = value
            else: self.data = value
            
    @ret_handler
    def __add__(self, other): return self.tensor + other.tensor
    @ret_handler
    def __sub__(self, other): return self.tensor - other.tensor 
    @ret_handler
    def __mul__(self, other): return self.tensor * other.tensor 
    def __repr__(self): return f"<Tensor of Shape: {self.shape} dtype> {self.dtype}>"
    @ret_handler
    def sadd(self, other): return self.tensor + other.tensor # Tensor + Scalar
    
    def set (self, value: Union[int, float, bool, dtype]=0): self.tensor = np.ndarray(self.shape, self.dtype).zero()

    
    