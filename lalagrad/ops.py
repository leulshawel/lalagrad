#How do you want the result of your object and some op assertions
def binary_op_wrapper(func):
        def wrapper(self, other):
            #assert requirements for binary ops
            assert self.shape == other.shape and self.device == other.device and \
                self.data is not None, "Improper Tensors for Operation"
            _class = self.__class__
            #new Tensor with the result of the op
            strongest_dtype = self.dtype if self.dtype.strength > other.dtype.strength else other.dtype
            new = _class(data=None, shape=self.shape, dtype=strongest_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
            new.data = func(self, other) 
            if self.strong and other.strong: return  new #return new if both are strong tensors
            if self.strong: other.data, other.dtype = func(self, other), strongest_dtype
            else: self.data, self.dtype = func(self, other), strongest_dtype  
        return wrapper
    
    
#samething but for unary ops
def unary_op_wrapper(func):
    def wrapper(self, other):
        assert self.data is not None, "Op on Empty Tensor"
        _class = self.__class__
        new = _class(data=None, device=self.device, requires_grad=self.requires_grad) 
        new.data = func(self, other)
        if self.strong: return new
        self.data = func(self, other)
        self.dtype = self.dtype if self.dtype.strength > new.dtype.strength else new.dtype
    return wrapper