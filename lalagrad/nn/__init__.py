from typing import List, Union
from math import e

#Activation
class Acts:
    @staticmethod
    def Relu(l: List[Union[int, float]]): return [x if x > 0 else 0 for x in l]
    @staticmethod
    def Sigmoid(l: List[Union[int, float]]): return [1/(1 + e**-x) for x in l]
    @staticmethod
    def Tanh(l: List[Union[int, float]]): return [(e**x - e**-x) / (e**x + e**-x) for x in l]
    @staticmethod
    def LeakyRelu(l: List[Union[int, float]], alpha=0.01):return [x if x > 0 else alpha*x for x in l]
    @staticmethod
    def Signum(l: List[Union[int, float]]):return [1 if x > 0 else -1 for x in l]
    @staticmethod
    def Softmax(l: List[Union[int, float]]):
        height = sum([e**x for x in l])
        return [e**x / height for x in l]