from enum import Enum, auto
from dataclasses import dataclass

@dataclass(frozen=True)
class Benchs:
    ELWISE = auto(); MATMUL = auto(); REDUCE_MUL = auto()
    
