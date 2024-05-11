<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b>This will be a mid-level Deep learning framework</b><br>
<b>For Educational purposes mainly</b><br>




<b>Tensors</b>
```python
from lalagrad.tensor import Tensor

x = Tensor(data=[[3, 4, 5]])
y = Tensor(data=[[10, 20, 30]])

#you wanna add the two
z = x + y

print(z.tolist())

#you wanna add the two and save the result in one of them
with x(): #or   with y(): to save the result on y
    x + y

print(x.tolist())
```