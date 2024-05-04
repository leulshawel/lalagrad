import unittest, sys, os
from lalagrad.tensor import Tensor 

class TestTensorInit(unittest.TestCase):
    def setUp(self): 
       self.t1 = Tensor([[1, 22]])
       self.t2 = Tensor([[True, False]])
       
    def test_dtype(self): self.assertEqual(self.t1.dtype.eq, self.t1.data[0].__class__)

if __name__ == "__main__":
   unittest.main()