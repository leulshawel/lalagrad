
from lalagrad import Tensor
from typing import Union, List, Tuple

class FCFF:
    def __init__(self, layers: Union[List, Tuple]):
        self.weights = [Tensor.rand(shape=(layers[i-1], layers[i])) for i in range(1, len(layers))]
        self.biases = [Tensor.rand(shape=(1, layers[i])) for i in range(1, len(layers))]
        
    def forward(self, x: Tensor):
        for l, b in zip(self.weights, self.biases):
            x = x.matmul(l)
            x.Relu()
            x += b
        return x
    

if __name__ == "__main__":
    #Model
    inp , hidd, out = [3], [2, 4, 3, 4], [2]
    nn = FCFF(inp + hidd + out)
    #Input data
    x = Tensor([[1, 0, 1]])
    #Feed forward
    r = nn.forward(x)
    
    print(r.tolist()) 
    
