import unittest, math
import numpy as np
from lalagrad.tensor import Tensor 


class TestTensorOps(unittest.TestCase):
    def setUp(self): 
       self.x = Tensor([[100.0, 100.0]])
       self.y = Tensor([[22, 1]])
       print(self.x)
       
       self.a = np.array([[100.0, 100.0]])       
       self.b = np.array([[22, 1]])       
       
    def test_add(self): self.assertEqual((self.x + self.y).tolist(), (self.a + self.b).tolist())
    def test_sub(self): self.assertEqual((self.x - self.y).tolist(), (self.a - self.b).tolist())
    def test_mul(self): self.assertEqual((self.x * self.y).tolist(), (self.a * self.b).tolist())
    #def test_log(self): self.assertEqual((self.x + self.y).tolist(), (self.a + self.b).tolist())
    
    #reduce ops
    def test_sum(self): pass
        
        
    
if __name__ == "__main__":
    unittest.main()