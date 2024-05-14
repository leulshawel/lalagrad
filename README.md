<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b>lalagrad</b> will be a mid-level Deep learning framework for Educational purposes mainly
between tinygrad and a micrograd

though am building this to learn, the final thing has to be as fast as <b>Numpy</b> if not faster

<br><br>
<b>Tensors</b><br>
most tensor creation methods and operations supported by [@tinygrad](https://github.com/tinygrad/tinygrad) will be here (may be all of them) look at lalagrad/tensor.py

```python
from lalagrad import Tensor

x = Tensor.eye(3)       #identity matrice of shape (3, 3)

#from np ndarray
import numpy as np
y = Tensor(np.array([[2.0, 0, -2.0]]))

y = y.transpose()
z = x.matmul(y)

#reshape
y.reshape((3, 1))

#if u wanna put the returns of your ops on one of the operands
with x():   #do with y(): to put the result on y
    x + z
    
print(x.sum())
print(y.sum(axis=0).tolist())
print(z.max(axis=1).tolist())                  
```

<br><b>Feed Forward </b>with no backprob

```python
#at this point we can implement a simple fully connected feed forward nn without backprop

from lalagrad import Tensor
from typing import Union, List, Tuple

class FCFF:
    def __init__(self, layers: Union[List, Tuple]):
        self.weights = [Tensor.rand(shape=(layers[i-1], layers[i])) for i in range(1, len(layers))]
        self.biases = [Tensor.rand(shape=(1, layers[i])) for i in range(1, len(layers))]
    def forward(self, x: Tensor):
        for l, b in zip(self.weights, self.biases):
            x = x.matmul(l) + b
        return x
    
if __name__ == "__main__":
    #Model
    nn = FCFF([3, 2, 1])

    x = Tensor([[1, 0, 1]])
    r = nn.forward(x)
    
    print(r.tolist()) 
```
