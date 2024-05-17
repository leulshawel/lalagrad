from bench.lalagrad.funcs import *
from bench import Benchs
from statistics import mean

class SpeedBench:
    def __init__(self, test): self.test = test
    def __call__(self, loop=10000, avg=1): 
        if self.test == Benchs.ELWISE:   SpeedBench.elwise(loop, avg)
        elif self.test == Benchs.MATMUL: SpeedBench.matmul(loop, avg)
        elif self.test == Benchs.REDUCE_MUL: SpeedBench.reduce_mul(loop, avg)
        
    @staticmethod
    def matmul(loop, avg):
        lala, np = mean([lala_matmul(loop) for _ in range(avg)]), mean([np_matmul(loop) for _ in range(avg)])
        r = lala / np 
        print(f"lala: {lala}\nnumpy: {np}\nabout {r} times {'slower' if r > 1 else 'faster' }")
        
    @staticmethod
    def elwise(loop, avg):
        lala, np = mean([lala_elwise(loop) for _ in range(avg)]), mean([np_elwise(loop) for _ in range(avg)])
        r = lala / np 
        print(f"lala: {lala}\nnumpy: {np}\nabout {r} times {'slower' if r > 1 else 'faster' }")
        
    @staticmethod
    def reduce_mul(loop, avg):
        lala, np = mean([lala_mul(loop) for _ in range(avg)]), mean([np_mul(loop) for _ in range(avg)])
        r = lala / np 
        print(f"lala: {lala}\nnumpy: {np}\nabout {r} times {'slower' if r > 1 else 'faster' }")
    
    
        
    