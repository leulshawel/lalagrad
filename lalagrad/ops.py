#How do you want the result of your object and some op assertions
def binary_op_wrapper(to_tensor=True):
    def decorator(func):
        def wrapper(self, other):
            _class = self.__class__
            if to_tensor and not isinstance(other, _class): other = _class.full(val=other, shape=self.shape)
            #assert requirements for binary ops
            assert self.shape == other.shape and self.dtype is not None and other.dtype is not None and self.data is not None and other.data is not None, "Improper Tensors for Operation"
            #new Tensor with the result of the op
            strongest_dtype = self.dtype if self.dtype > other.dtype else other.dtype
            data, shape = func(self, other)
            new = _class(data=None, shape=shape, dtype=strongest_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
            if self.strong and other.strong: 
                new.setdata(data)
                return  new #return new if both are strong tensors
            if self.strong: other.data, other.dtype = data, strongest_dtype
            else: self.data, self.dtype = data, strongest_dtype  
        return wrapper
    return decorator
    
    
#samething but for unary ops
def unary_op_wrapper(to_tensor=True):
    def decorator(func):
        def wrapper(self, other=None):
            _class = self.__class__
            if to_tensor and not isinstance(other, _class): other = _class.full(val=other, shape=self.shape)
            assert self.data is not None, "Op on Empty Tensor"
            data, shape = func(self, other)
            new = _class(data=None, shape=shape, device=self.device, requires_grad=self.requires_grad) 
            new.dtype = (self.dtype if self.dtype > new.dtype else new.dtype) if new.dtype is not None else self.dtype
            new.setdata(data)
            if self.strong: return new
            self.data = data
        return wrapper  
    return decorator