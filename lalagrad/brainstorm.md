lalagrad:
        - is inspired and guided by George Hotz's tinyrgrad
        - plans to use numba for speed
        - uses lists a unit of data storage and try to Never use appends but build every list in a single run 

implement a small Tensor class
        - requires multidimentional data or shape
        - a tensor is just a list (the data is stored in a single fully flattened list)
        - a shape is just a way to look at it
        - a tensor could be strong [No op changes its value] or Weak [every op changes it value]
        - the result of a reduce op in lalagrad is a new tensor of the same dimension. 
        Ex: a reduce op along the second dimention[axis=1] of a tensor of shape(3, 4, 5) is of shape (3, 1, 5) unlike numpy which returns an array of shape(3, 5)
        - you can't have a tensor of shape(n,) in lalagrad. the smallest dimension tensor is a Matrice (2D). a vector is just a row matrice (n, 1) or a column matrice (1, n) and a scalar is a (1, 1) Tensor
        - transpose and matmul only defined for Matrices (2D Tensors)

accelerate Ops
        - We need a n conputation graph
                maybe interaction conbinators 
                Operation graphs
        - Numba 
        - Multprocessing
                go through the operations and parallelize independent ops
                take a single tensor op and parallelize ops along the dimensions or elements