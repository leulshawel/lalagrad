<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b>lalagrad</b> will be a mid-level Deep learning framework for Educational purposes mainly
between tinygrad and a micrograd

<br><br>
<b>Tensors</b>

```python
from lalagrad.tensor import Tensor



x = Tensor(data=[[3, 4, 5, 1]]) # x.shape => (1, 3)

 # Create from numoy ndarray
import numpy as np
array = np.array([[10, 12, 14, 2]])
y = Tensor(data=array) 


y_like = y.zeros_like()         # create a tensor like y but filled_with zeros
identity = Tensor.eye(4)        #identity matrice

#you wanna add the two
z = x + y

#you wanna add the two and save the result in one of them
with x(): #or with y(): to save the result on y
    x + y

#reshape
x.reshape((2, 2))

#reduce (axis wise ops)  these always returns a new tensor (they never modify self)       
m = x.max(axis=1)                        

```