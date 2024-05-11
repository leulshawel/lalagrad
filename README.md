<img style="float: left" src=./lalagrad/utils/img/lala.jpeg alt=drawing width=200/>
<b>This will be a mid-level Deep learning framework</b></br>
<br><br><b>For Educational purposes mainly</b><br>

If your professor tells you to train a classifier for the next class<br>
or you want a framework you wanna read and undersand in a few hours<br>
THIS IS FOR YOU

utill i deside to make it big...


Tensors
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