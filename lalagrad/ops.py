#How do you want the result of your object and some op assertions
def binary_op_wrapper(to_tensor=True, equal_shape=True):
    def decorator(func):
        def wrapper(self, other):
            _class = self.__class__
            if to_tensor and not isinstance(other, _class): other = _class.full(val=other, shape=self.shape)
            #assert requirements for binary ops
            assert self.dtype is not None and other.dtype is not None and self.data is not None and other.data is not None, "Improper Tensors for Operation"
            if equal_shape: assert self.shape == other.shape, "Tensors are not of the same shape"
            strongest_dtype = self.dtype if self.dtype > other.dtype else other.dtype
            data, shape = func(self, other)
            if self.strong and other.strong: 
                #new Tensor with the result of the op
                new = _class(data=None, shape=shape, dtype=strongest_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
                new.setdata(data)
                #return new if both are strong tensors
                return  new 
            if self.strong: 
                other.dtype = strongest_dtype
                other.setdata(data)
            else: 
                self.dtype = strongest_dtype  
                self.setdata(data)
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
            if self.strong: 
                new = _class(data=None, shape=shape, device=self.device, requires_grad=self.requires_grad) 
                new.dtype = (self.dtype if self.dtype > new.dtype else new.dtype) if new.dtype is not None else self.dtype
                new.setdata(data)
                return new
            self.shape = shape
            self.setdata(data)
        return wrapper  
    return decorator

    