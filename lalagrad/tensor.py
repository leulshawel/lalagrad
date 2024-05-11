import numpy as np
from typing import Optional, Union, List
import math, numpy

from lalagrad.dtype import DType, dtypes, TYPES_DICT
from lalagrad.device import Device ,devices
from lalagrad.ops import binary_op_wrapper, unary_op_wrapper
from lalagrad.array_ops import flatten, array_from_shape, add_const,\
    shape_from_array, reverse, scale, devide_array, _set, rand_array_from_shape,\
    map_along_axis, build_higher_dim
    

        
class Tensor():
    __slots__ = "data", "device", "shape", "dtype", "ctx", "strong", "requires_grad", "grad"
    
    def __init__(self, data: Union[None, List, np.ndarray]=None, shape: tuple[int]=None, dtype: Optional[DType]=None, 
                 device: Device=devices.CPU, ctx = None, requires_grad=False, strong: bool=True):
        assert data is not None or shape is not None, "Tensor object requires atleast a data or a shape"
        if data is None: self.data, self.shape, self.dtype = None, shape, None
        elif isinstance(data, np.ndarray): 
            data, self.dtype = data.tolist(), TYPES_DICT[data.dtype.name] 
            self.data, self.shape = flatten(data), tuple(reverse(shape_from_array(data)))                                                                                       
        elif isinstance(data, (list, tuple)):   
            assert (all([isinstance(r, (list, tuple)) for r in data]) or shape) and len(data), "improper data" #Scalars and Vectors are not supported yet
            self.data, self.shape = flatten(data), tuple(reverse(shape_from_array(data)))
            self.dtype =  next((v for  v in TYPES_DICT.values() if v.eq == self.data[0].__class__), None) if dtype is None else dtype 

        self.device, self.strong, self.ctx, self.requires_grad = device, strong, ctx, requires_grad
        self.grad: Optional[Tensor] = None
        
    @classmethod
    def zeros(cls, shape, dtype=dtypes.int16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): 
        return cls(array_from_shape(shape, 0), shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def ones(cls, shape, dtype=dtypes.int16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): 
        return cls(array_from_shape(shape, 1), shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def full(cls, val, shape, dtype=dtypes.int16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): 
        return cls(array_from_shape(shape, val), shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def empty(cls, shape, dtype=dtypes.int16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): 
        return cls(None, shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def rand(cls, shape, dtype=dtypes.float16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): 
        return cls(rand_array_from_shape(shape), shape, dtype, device, ctx, requires_grad, strong)
    @classmethod
    def eye(cls, rows, colns = None, dtype=dtypes.int16, device=devices.CPU, ctx=None, strong=None, requires_grad=False): 
        if colns is None: colns = rows
        data = [[1 if j==i else 0 for j in range(colns)] for i in range(rows)]
        return  cls(data, dtype=dtype, device=device, ctx=ctx, requires_grad=requires_grad, strong=strong)
    def ones_like(self): 
        return Tensor.ones(self.shape, self.dtype, self.device, self.ctx, self.requires_grad, self.strong)
    def zeros_like(self): 
        return Tensor.zeros(self.shape, self.dtype, self.device, self.ctx, self.requires_grad, self.strong)
    
    
    def __call__(self): return self
    def __enter__(self): self.strong = False 
    def __exit__(self, *args): self.strong = True
    
    #get the data with right dimension (unflatten)
    def tolist(self, l=None, n=0):
        assert self.data is not None, "view() on empty tensor with no dimension"
        if not l: l = self.data
        if n+1 == len(self.shape): return l
        return [self.tolist(dl, n+1) for dl in devide_array(l, self.shape[n])]  
    
    #get Tensor properties
    def is_float(self): return self.dtype in (dtypes.float16, dtypes.float32, dtypes.float64)
    def get_device(self): return self.device  
    def numel(self): return math.prod(self.shape)  
    def __repr__(self): return f"<Tensor: of shape: {self.shape}, {self.dtype} on {self.device} with grad: {self.grad}>"
    
    #set tensor properties
    def set_device(self, d: Device): self.device = d
            
    #on self or return binary ops
    #TODO: what about device?
    def __eq__(self, other): return self.data == other.data and self.shape == other.shape 
    @binary_op_wrapper  
    def __add__(self, other): return [x+y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __sub__(self, other): return [x-y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __mul__(self, other): return [x*y for x, y in zip(self.data, other.data)]
    @binary_op_wrapper
    def __truediv__(self, other): return [(round(x/y, self.dtype.precision) if self.dtype > other.dtype else round(x/y, other.dtype.precision)) if self.dtype is not None and other.dtype is not None else x/y for x, y in zip(self.data, other.data)]
    def __eq__(self, other): return self.dtype == other.dtype and self.shape == other.shape
    def dot(self, other):
        assert len(self.shape) == len(other.shape) == 2 and (1 in self.shape and 1 in other.shape), "dot is defined only for vectors"
        return sum([x*y for x, y in zip(self.data, other.data)])
    #TODO: matrix mutiplication
    def matmul(self, other): pass 
    def tensordot(self, other, dims): pass
    
    #on self or return unaryOps
    @unary_op_wrapper
    def sadd(self, s): return add_const(self.data, s)
    @unary_op_wrapper
    def smul(self, s): return scale(self.data, s)
    @unary_op_wrapper
    def __not__(self):  return [not b for b in self.data] if self.dtype==DType('bool') else [-1 * e for e in self._dat]
    @unary_op_wrapper
    def __pow__(self, e):  
        return  [elem**e for elem in self.data] if self.dtype not in (dtypes.bool,)  else None
    @unary_op_wrapper
    def log(self, b=10): 
        assert b != 1, "base can't be One"
        return [round(math.log10(elem)/math.log10(b), self.dtype.precision) if self.dtype.precision is not None else math.log10(elem)/math.log10(b) for elem in self.data]
    
    
    #on self ops
    def check(self):
        assert all([e.__class__ == self.dtype.eq for e in self.data]), "check failed on dtype" 
        assert len(self.data) == math.prod(self.shape), "check failed on shape"
        print("Tensor object in optimal state")
        
    def setdata(self, val: Union[int, float, bool]): self.data = _set(self.data, val)
    #TODO: expand the tensor in a axis
    def expand(self, shape, n=0): pass
    def transpose(self, axis1, axis2): self.shape[axis1], self.shape2 = self.shape[axis2], self.shape[axis1]
    def reshape(self, shape):
        assert math.prod(self.shape) == math.prod(shape), "Tensor can't be of this shape"
        self.shape = tuple(shape)
    
    #reduce ops
    #add elemens in a single axis
    def sum(self, axis=None):
        if axis is None: return max(self.data)
        assert axis < len(self.shape), "dimension doesn't exist"
        reduced = Tensor(build_higher_dim(axis, self.tolist(), sum)) if axis != 0 else Tensor([map_along_axis(self.tolist(), sum)])
        reduced.shape = tuple(1 if i==axis else e for i, e in enumerate(self.shape))
        return reduced
    #mul elemens in a single axis
    def mul(self, axis=None):
        if axis is None: return max(self.data)
        assert axis < len(self.shape), "dimension doesn't exist"
        reduced = Tensor(build_higher_dim(axis, self.tolist(), math.prod)) if axis != 0 else Tensor([map_along_axis(self.tolist(), math.prod)])
        reduced.shape = tuple(1 if i==axis else e for i, e in enumerate(self.shape))
        return reduced    #min along an axis or of a Tensor
    def min(self, axis=None):
        if axis is None: return max(self.data)
        assert axis < len(self.shape), "dimension doesn't exist"
        reduced = Tensor(build_higher_dim(axis, self.tolist(), min)) if axis != 0 else Tensor([map_along_axis(self.tolist(), max)])
        reduced.shape = tuple(1 if i==axis else e for i, e in enumerate(self.shape))
        return reduced
    #max along an axis or of a Tensor
    def max(self, axis=None): 
        if axis is None: return max(self.data)
        assert axis < len(self.shape), "dimension doesn't exist"
        reduced = Tensor(build_higher_dim(axis, self.tolist(), max)) if axis != 0 else Tensor([map_along_axis(self.tolist(), min)])
        reduced.shape = tuple(1 if i==axis else e for i, e in enumerate(self.shape))
        return reduced                  