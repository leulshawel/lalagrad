#How do you want the result of your object
def binary_op_wrapper(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.strong and other.strong: return self.new(data=value, shape=self.shape, device=self.device, 
                dtype=self.dtype if self.dtype.strength > other.dtype.strength else other.dtype, requires_grad=self.requires_grad or other.requires_grad) 
            if self.strong: other.data = value; other.dtype = self.dtype if self.dtype.strength >= other.dtype.strength else other.dtype
            else: self.data = value; self.dtype = self.dtype if self.dtype.strength > other.dtype.strength else other.dtype  
        return wrapper  
    

def unary_op_wrapper(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.strong: return self.new(data=value, requires_grad=self.requires_grad) 
            self.data = value
        return wrapper