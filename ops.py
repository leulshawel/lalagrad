#How do you want the result of your object
def binary_op_wrapper(func):
        def wrapper(self, other):
            assert self.shape==other.shape and self.dtype.dtype==other.dtype.dtype, f"can't Op {self} with {other}"
            value = func(self, other)
            if self.strong and other.strong: return self.new(data=value)
            if self.strong: other.data = value
            else: self.data = value
        return wrapper
    

def unary_op_wrapper(func):
        def wrapper(self, other):
            value = func(self, other)
            if self.strong: return self.new(data=value)
            self.data = value
        return wrapper