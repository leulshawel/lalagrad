<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b><h1>lalagrad</h1></b> lalagrad is a mid-level Deep learning framework currently under development<br> 

for Educational purposes mainly and will be between [@tinygrad](https://github.com/tinygrad/tinygrad) and [@microrad](https://github.com/karpathy/micrograd).

<h3><b>The plan</b></h3>
Though am building this to learn, the final thing has to be as fast as <b>Numpy</b> if not faster.
And with a decent nn module<br>

<h3><b>Tensors</b></h3>

most tensor creation methods and operations supported by tinygrad will be here (may be all of them) look at lalagrad/tensor.py

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

<h3><b>Benchs</b></h3>

currently we are about around 15 times slower than Numpy for matmul operations and 5 times slower for element wise ops

```python
from bench.lalagrad.speed import SpeedBench, Benchs
#average a 100 runs with 10000 matmul ops in each loop
SpeedBench(test=Benchs.MATMUL)(loop=10000, avg=100)  
```
