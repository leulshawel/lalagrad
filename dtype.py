from typing import Final
from dataclasses import dataclass

@dataclass(order=True, frozen=True)
class DType:
    strength: int #higher strength type will be the type of op results
    name: str #name of the data type
    bytes: int #How many bytes does it take
    sz: int    #
    def __repr__(self): return f"{ self.dtype }"
    
  
#Common data types  
class dtypes:
  @staticmethod
  def get_type(val): return dtypes.bool if isinstance(val, (bool)) else (dtypes.int8)
  bool: Final[DType]     = DType(0, 1, "bool", '?', 1)
  int8: Final[DType]     = DType(1, 1, "char", 'b', 1)
  uint8: Final[DType]    = DType(2, 1, "unsigned char", 'B', 1)
  int16: Final[DType]    = DType(3, 2, "short", 'h', 1)
  uint16: Final[DType]   = DType(4, 2, "unsigned short", 'H', 1)
  int32: Final[DType]    = DType(5, 4, "int", 'i', 1)
  uint32: Final[DType]   = DType(6, 4, "unsigned int", 'I', 1)
  int64: Final[DType]    = DType(7, 8, "long", 'l', 1)
  uint64: Final[DType]   = DType(8, 8, "unsigned long", 'L', 1)
  float16: Final[DType]  = DType(9, 2, "half", 'e', 1)
  bfloat16: Final[DType] = DType(10, 2, "__bf16", None, 1)
  float32: Final[DType]  = DType(11, 4, "float", 'f', 1)
  float64: Final[DType]  = DType(12, 8, "double", 'd', 1)
  