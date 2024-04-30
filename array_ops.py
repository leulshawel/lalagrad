#some usable funcs and classes no directly related to any object

from typing import List, Union, Tuple
from dtype import Dtype


def get_shape(data: List[Union[int, float, bool, List]]): pass
#from tinygrad.tensor
def flatten(l: Union[List, Tuple]): return [item for sublist in l for item in (flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
#generate an array from a shape
def array_from_shape(shape: Union[List, tuple, int], val=0): return [val for _ in range(shape[0])] if len(shape)==1 else [array_from_shape(shape[1:], val) for _ in range(shape[0])]
#add a constant value on every element of a list
def add_const(l: Union[List, Tuple], c: Union[int, float, bool, Dtype]): return [add_const(e, c) if isinstance(e, (list, tuple)) else e+c for e in l]
#set every element to a value
def set (l: Union[List, Tuple], val: Union[int, float, bool, Dtype]=0): return [set(e, val) if isinstance(e, (list, tuple)) else val for e in l]
