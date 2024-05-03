import numpy as np
from typing import Optional, Union, List

from lalagrad.dtype import DType, dtypes
from lalagrad.device import Device ,devices
from lalagrad.ops import binary_op_wrapper, unary_op_wrapper
from lalagrad.array_ops import flatten, array_from_shape, add_const,\
    shape_from_array, reverse, num_of_elems, scale, devid_list, _set

class Tensor():
    __slots__ = "data", "device", "shape", "dtype", "ctx", "strong", "mat", "requires_grad", "grad"
    class Matrix:
        #TODO: implement matrix multiplication 
        def matmul(self, other):
            return [[0]]
    
    def __init__(self, data: Optional[Union[None, List, int, float, bool]], shape: tuple[int]=None, dtype: Optional[DType]=None, 
                 device: Device=devices.CPU, ctx = None, requires_grad=False, strong: bool=True):
        assert (all([isinstance(r, (list, tuple)) for r in data]) or shape) and len(data), "improper data"
        
        self.data, self.shape = flatten(data), tuple(reverse(shape_from_array(data)))
        self.dtype, self.device, self.strong, self.ctx, self.requires_grad = dtype, device, strong, ctx, requires_grad
        self.grad: Optional[Tensor] = None
        self.mat = Tensor.Matrix() if len(self.shape) == 2 else None
        
    @classmethod
    def new(cls, data, shape, dtype=None, device=devices.CPU, ctx=None, strong=None, requires_grad=False): return cls(data, shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def zeros(cls, shape, dtype=dtypes.int16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): return cls(array_from_shape(shape, 0), shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def ones(cls, shape, dtype=dtypes.int16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): return cls(array_from_shape(shape, 1), shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def ones_like(cls, self): return cls.ones(self.shape, self.dtype, self.device, self.ctx, self.requires_grad, self.strong)
    
    #get the data with right dimension (unflatten)
    def view(self, l=None, n=0): 
        if not l: l = self.data
        if n+1 == len(self.shape): return l
        return [self.view(dl, n+1) for dl in devid_list(l, self.shape[n])]  
    
    #get Tensor properties
    def is_float(self): return self.dtype in (dtypes.float16, dtypes.float32, dtypes.float64)
    def get_device(self): return self.device  
    def numel(self): return num_of_elems(self.shape)  
    def __repr__(self): return f"<Tensor: of shape: {self.shape}, {self.dtype} on {self.device} with grad: {self.grad}>"
    
    #set tensor properties
    def set_device(self, d: Device): self.device = d
            
    #on self or return binary ops
    @binary_op_wrapper  
    def __add__(self, other): return [x+y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __sub__(self, other): return [x-y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __mul__(self, other): return [x*y for x, y in zip(self.data, other.data)]
    def matmul(self, other):  
        assert self.mat is not None and other.mat is not None,\
        "matrix multiplication only for 2D Tensors (Matrices)"
        return Tensor(data=self.mat.matmul(other.mat))
    
    #Order
    def __eq__(self, other): return self.dtype == other.dtype and self.shape == other.shape
    
    #on self or return unaryOps
    @unary_op_wrapper
    def sadd(self, s): return add_const(self.data, s)
    @unary_op_wrapper
    def smul(self, s): return scale(self.data, s)
    @unary_op_wrapper
    def __pow__(self, e):  
        return  [elem**e for elem in self.data] if self.dtype not in (dtypes.bool,)  else None
    @unary_op_wrapper
    def __not__(self):  return [not b for b in self.data] if self.dtype==DType('bool') else []
    
    #on self ops
    def set_data(self, val: Union[int, float, bool]): self.data = _set(self.data, val)
    #TODO: expand the tensor in a axis
    def expand(self, shape, n=0): pass             
    def transpose(self, axis1, axis2): self.shape[axis1], self.shape2 = self.shape[axis2], self.shape[axis1]
    def reshape(self, shape):
        assert num_of_elems(self.shape) == num_of_elems(shape), "Tensor can't be of this shape"
        self.mat = Tensor.Matrix() if len(shape) == 2 else None
        self.shape = tuple(shape)
    
    #TODO: reduce Ops  
    #dot product
    def dot(self, other, axis): pass
    #add elemens in a single axis
    def sum(self, axis): return 
    #elemnt-wise multiplication
    def mul_aix(self, other): return self * other
    #min along an axis or of a Tensor
    def min(self, axis): pass
    #max
    def max(self, axis): pass