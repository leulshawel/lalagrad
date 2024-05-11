#some usable funcs and classes not directly related to any object
from typing import List, Union, Tuple
from lalagrad.dtype import DType
import random, math



def get_shape(data: List[Union[int, float, bool, List]]): pass
#from tinygrad.tensor
def flatten(l: Union[List, Tuple]): return [item for sublist in l for item in (flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
#generate an array from a shape
def array_from_shape(shape: Union[List, tuple, int], val=0): return [val for _ in range(shape[0])] if len(shape)==1 \
    else [array_from_shape(shape[1:], val) for _ in range(shape[0])]
#random vals
def rand_array_from_shape(shape: Union[List, tuple, int], p=4): return [round(random.random(), p)for _ in range(shape[0])] if len(shape)==1 \
    else [rand_array_from_shape(shape[1:], p) for _ in range(shape[0])]
#get shape from array assuming you provided the right tensor and return is in reverse order
def shape_from_array(l: Union[List, Tuple]): return shape_from_array(l[0]) + [len(l)] if isinstance(l[0], (tuple, list)) else [len(l)]
#add a constant value on every element of a list
def add_const(l: Union[List, Tuple], c: Union[int, float, bool, DType]): return [add_const(e, c) if isinstance(e, (list, tuple)) else e+c for e in l]
#multiply with a scalar
def scale(l: Union[List, Tuple], c: Union[int, float, bool, DType]): return [scale(e, c) if isinstance(e, (list, tuple)) else e*c for e in l]
#set every element to a value
def _set (l: Union[List, Tuple], val: Union[int, float, bool, DType]=0): return [_set(e, val) if isinstance(e, (list, tuple)) else val for e in l]

#normal list.reverse() doesn't return
def reverse(l): 
    l.reverse()
    return l

#devide a list in to n number if lists of equal size
def devide_array(l: Union[List, Tuple], n: int):
    assert not len(l) % n, f"array not devisible in to equal {n} arrays"
    ne = int(len(l)/n)
    return [l[i*ne:ne*(i+1)] for i in range(n)]

#map a function on corresponding elements along a dimentsion
def map_along_axis(l: Union[Tuple, List], f):
    _class = l[0].__class__
    assert all([e.__class__ == _class for e in l])
    if isinstance(l[0], (int, float)): return [f(l)]
    shape = shape_from_array(l[0]) 
    assert all([shape_from_array(sub_l) == shape for sub_l in l])
    return [f([ll[i] for ll in l]) for i in range(len(l[0]))] if isinstance(l[0][0], (int, float)) else [map_along_axis([ll[i] for ll in l], f) for i in range(len(l[0]))]
#builds all the higher dimensions of the a tensor and pults the results of a lower dimension op in them
def build_higher_dim(axis, l, f, d=1):
        return [map_along_axis(ll, f) if d == axis else build_higher_dim(axis, ll, f, d+1) for ll in l]

    
