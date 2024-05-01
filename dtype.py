class Dtype:
    int32 = int
    float = float
    def __init__(self, data=None, dtype =None): self.dtype = dtype if dtype is not None else data.__class__
    def __repr__(self): return f"{ self.dtype }"