from bench.funcs import *
from bench import Benchs

class SpeedBench:
    def __init__(self, test): self.test = test
    def __call__(self):
        if self.test == Benchs.ELWISE: return SpeedBench.elwise()
        elif self.test == Benchs.MATMUL:  return SpeedBench.matmul()
        
    @staticmethod
    def matmul():
        lala, np = lala_matmul(), np_matmul()
        r = lala / np 
        print(f"lala: {lala}\nnumpy: {np}\nabout {r} {'slower' if r > 1 else 'faser' }")
    
    @staticmethod
    def elwise():
        lala, np = lala_elwise(), np_elwise()
        return lala, np, lala/np
        
    