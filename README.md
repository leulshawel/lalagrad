<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b>lalagrad</b> will be a mid-level Deep learning framework for Educational purposes mainly
<br>between tinygrad and a micrograd...closer to the tiny :)

<br><br>
<b>Tensors</b><br>
most tensor creation methods and operations supported by [@tinygrad](https://github.com/tinygrad/tinygrad)<br>will be here (may be all of them) look at lalagrad/tensor.py

```python
from lalagrad import Tensor

#there are different ways to create a tensor
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
