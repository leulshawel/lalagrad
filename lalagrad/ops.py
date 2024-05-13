#How do you want the result of your object and some op assertions
def binary_op_wrapper(func):
        def wrapper(self, other):
            _class = self.__class__
            if isinstance(other, (int, float, bool)): other = _class([[other]])
            #assert requirements for binary ops
            assert self.shape == other.shape and self.dtype is not None and other.dtype is not None and self.data is not None and other.data is not None, "Improper Tensors for Operation"
            #new Tensor with the result of the op
            strongest_dtype = self.dtype if self.dtype > other.dtype else other.dtype
            data, shape = func(self, other)
            new = _class(data=None, shape=shape, dtype=strongest_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
            new.data = data
            if self.strong and other.strong: return  new #return new if both are strong tensors
            if self.strong: other.data, other.dtype = func(self, other), strongest_dtype
            else: self.data, self.dtype = data, strongest_dtype  
        return wrapper
    
    
#samething but for unary ops
def unary_op_wrapper(func):
    def wrapper(self, other):
        assert self.data is not None, "Op on Empty Tensor"
        _class = self.__class__
        data, shape = func(self, other)
        new = _class(data=None, shape=shape, device=self.device, requires_grad=self.requires_grad) 
        new.data, new.dtype = data, (self.dtype if self.dtype > new.dtype else new.dtype) if new.dtype is not None else self.dtype
        if self.strong: return new
        self.data = data
    return wrapper  