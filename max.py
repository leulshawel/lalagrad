import math
from lalagrad.tensor import Tensor

data = [
    [
        [2, 1, 11, 23],
        [34, 10, 21, 203],
        [120, 91, 404, 2.2]
    ],
    [
        [12, 15, 131, 213],
        [2, 1, 141, 233],
        [122, 21, 444, 94]
    ]
]

t = Tensor(data)
