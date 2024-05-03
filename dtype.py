from typing import Final, Optional
from dataclasses import dataclass

@dataclass(order=True, frozen=True, repr=False)
class DType:
    strength: int #higher strength type will be the type of op results
    bytes: int #How many bytes does it take
    name: str #name of the data type
    fmt: Optional[str]
    
    def __repr__(self): return f"<dtype: {self.name}({self.fmt})>"
    
  
#Common data types  
class dtypes:
  @staticmethod
  def get_type(val): return dtypes.bool if isinstance(val, (bool)) else (dtypes.int8)
  #from geohotz tinygrad
  bool: Final[DType] = DType(0, 1, "bool", '?')
  int8: Final[DType] = DType(1, 1, "char", 'b')
  uint8: Final[DType] = DType(2, 1, "unsigned char", 'B')
  int16: Final[DType] = DType(3, 2, "short", 'h')
  uint16: Final[DType] = DType(4, 2, "unsigned short", 'H')
  int32: Final[DType] = DType(5, 4, "int", 'i')
  uint32: Final[DType] = DType(6, 4, "unsigned int", 'I')
  int64: Final[DType] = DType(7, 8, "long", 'l')
  uint64: Final[DType] = DType(8, 8, "unsigned long", 'L')
  float16: Final[DType] = DType(9, 2, "half", 'e')
  #bfloat16: Final[DType] = DType(10, 2, "__bf16", None)
  float32: Final[DType] = DType(11, 4, "float", 'f')
  float64: Final[DType] = DType(12, 8, "double", 'd')
  