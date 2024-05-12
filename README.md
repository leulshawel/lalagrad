<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b>lalagrad</b> will be a mid-level Deep learning framework for Educational purposes mainly
between tinygrad and a micrograd

<br><br>
<b>Tensors</b>

```python
from lalagrad.tensor import Tensor


#normal tensor creation
x = Tensor(data=[[3, 4, 5, 1]]) # x.shape => (1, 4)

 # Create from numpy ndarray
import numpy as np
array = np.array([[10, 12, 14, 2]])
y = Tensor(data=array) 
yt = y.transpose()                           # a column matrice (4, 1)
        

y_like = y.zeros_like()                     # create a tensor like y but filled_with zeros
identity = Tensor.eye(4)                    #identity matrice of (4, 4) shape
f = Tensor.full(val=5, shape=(3, 3))

#element wise Ops
z = x + y

#you wanna add the two and save the result in one of them
with x(): #or with y(): to save the result on y
    x + y
    
#on self Ops
x.reshape((2, 2))
yt.reshape((2, 2))

#matrice Ops 
z = x.matmul(yt)

#reduce OPs (axis wise ops)
m = z.max(axis=1)  
print(m.tolist())                      

```