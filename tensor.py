import numpy as np
from dtype import Dtype
from typing import Optional, Union, List


Scalar = Union[int, float, bool]

class Tensor():
    __slots__ = "shape", "dtype", "ctx", "ret" #No More attributes
    
    #I dont know if this is a good idea
    def ret_handler(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.ret and other.ret: return value
            elif self.ret: other.tensor = value
            else: self.tensor = value
    
    def __init__(self, data: Optional[Union[None, np.ndarray, List, Scalar]]=None, shape: tuple[int]=None, dtype: Optional[Dtype]=None, ctx = None, ret: bool=True):
        assert dtype == None or isinstance(dtype, Dtype), "Data type unknown"
        
        if isinstance(data, List) and data != None: 
            self.tensor = np.array(data)
            assert self.shape == None or self.shape == Tensor.get_shape(data), "Shape doesn't match"   
        else: 
            print("Unseported Dtype") 
            self.shape  = self.dtype = None
            
    # 
    def set (self, value=0):
        self.tensor = np.array(0)
    
    @ret_handler
    def __add__(self, other):
        return self.tensor + other.tensor #numpy addition
    
    @ret_handler
    def __sub__(self, other):
        return self.tensor - other.tensor #numpy subs
    
    @ret_handler
    def sadd(self, other): #Scalar addition
        assert isinstance(other, self.dtype), "incompatible datatype"
        return self.tensor + other.tensor
    
    
    def __repr__(self):
        return f"<Tensor> of <Shape>: {self.shape} <dtype>: {self.dtype}"
    
    @staticmethod
    def get_shape(data: List[Union[int, float, bool, List]]):
            shape = []
            
            return tuple(shape)