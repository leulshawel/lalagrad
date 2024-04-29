#some usable funcs and classes no directly related to any object

from typing import List, Union
from dtype import Dtype

Scalar = Union[int, float, bool]
dtype = type[Dtype]


def get_shape(data: List[Union[int, float, bool, List]]): pass
#from tinygrad.tensor
def flatten(l): return [item for sublist in l for item in (flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
