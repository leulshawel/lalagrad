#some usable funcs and classes not directly related to any object
from typing import List, Union, Tuple
from lalagrad.dtype import DType
import math


def get_shape(data: List[Union[int, float, bool, List]]): pass
#from tinygrad.tensor
def flatten(l: Union[List, Tuple]): return [item for sublist in l for item in (flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
#get the number of elements of an array from the shape
def num_of_elems(s: Union[Tuple, List]): return math.prod(s)  
#generate an array from a shape
def array_from_shape(shape: Union[List, tuple, int], val=0): return [val for _ in range(shape[0])] if len(shape)==1 \
    else [array_from_shape(shape[1:], val) for _ in range(shape[0])]
#get shape from array assuming you provided the right tensor and return is in reverse order
def shape_from_array(l: Union[List, Tuple]): return shape_from_array(l[0]) + [len(l)] if isinstance(l[0], (tuple, list)) else [len(l)]
#add a constant value on every element of a list
def add_const(l: Union[List, Tuple], c: Union[int, float, bool, DType]): return [add_const(e, c) if isinstance(e, (list, tuple)) else e+c for e in l]
#multipuly with a scalar
def scale(l: Union[List, Tuple], c: Union[int, float, bool, DType]): return [add_const(e, c) if isinstance(e, (list, tuple)) else e*c for e in l]
#set every element to a value
def _set (l: Union[List, Tuple], val: Union[int, float, bool, DType]=0): return [set(e, val) if isinstance(e, (list, tuple)) else val for e in l]
#normal list.reverse() doesn't return
def reverse(l): 
    l.reverse()
    return l

#devide a list in to n number if lists of equal size
def devid_list(l, n):
    assert not len(l) % n, f"array not devisible in to equal {n} arrays"
    ne = int(len(l)/n)
    return [l[i*ne:ne*(i+1)] for i in range(n)]
