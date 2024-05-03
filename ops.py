#How do you want the result of your object
def binary_op_wrapper(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.strong and other.strong: 
                new = self.new(data=value) ; new.dtype = self.dtype if self.dtype.strength > other.dtype.strength else other.dtype
                return new
            if self.strong: other.data = value; other.dtype = self.dtype if self.dtype.strength > other.dtype.strength else other.dtype
            else: self.data = value; self.dtype if self.dtype.strength > other.dtype.strength else other.dtype  
        return wrapper
    

def unary_op_wrapper(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.strong: return self.new(data=value)
            self.data = value
        return wrapper