<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b><h1>lalagrad</h1></b> lalagrad is a mid-level Deep learning framework currently in development<br> 
for Educational purposes mainly and will be between tinygrad and micrograd. as close as to the tiny as it needs to be

<h3><b>The plan</b></h3>
Though am building this to learn, the final thing has to be as fast as <b>Numpy</b> if not faster, with an nn module<br>

<h3><b>Tensors</b></h3>

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

<br>
<h3><b>Feed Forward </b></h3>

with no backprob. take a look at examples/fcff.py


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

<h3><b>Benchs</b></h3>

currently we are about around 15 times slower than Numpy for matmul operations and 5 times slower for element wise ops

```python
from bench.speed import SpeedBench, Benchs

if __name__ == "__main__":
    SpeedBench(test=Benchs.MATMUL)()
```
