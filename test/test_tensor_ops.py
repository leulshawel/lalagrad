import unittest, math
import numpy as np
from lalagrad.tensor import Tensor 

x = Tensor([[[100.0, 100.0], [44.0, 12.0]], [[77.0, 21.0], [44.0, 12.0]]])
m = Tensor([[100.0, 100.0], [44.0, 12.0]])

y = Tensor([[[44.0, 12.0], [77.0, 21.0]], [[100.0, 100.0], [44.0, 12.0]]])
       
b = np.array([[[44.0, 12.0], [77.0, 21.0]], [[100.0, 100.0], [44.0, 12.0]]])       
a = np.array([[[100.0, 100.0], [44.0, 12.0]], [[77.0, 21.0], [44.0, 12.0]]])  
c = np.array([[100.0, 100.0], [44.0, 12.0]])
     

class TestElementWiseOps(unittest.TestCase):
       
       
    def test_add(self): self.assertEqual((x + y).tolist(), (a + b).tolist())
    def test_sub(self): self.assertEqual((x - y).tolist(), (a - b).tolist())
    def test_mul(self): self.assertEqual((x * y).tolist(), (a * b).tolist())
    #def test_log(self): self.assertEqual((x + y).tolist(), (self.a + self.b).tolist())
    
class TestReduceOps(unittest.TestCase):
    #reduce ops
    def test_sum(self):
        r = x.sum(1)
        r.reshape((2, 2)) #this is needed cause numpy returns a (2, 2) matrice while lalagrad (2, 1, 2) so we reshape it to be (2, 2)
        self.assertEqual(r.tolist(), a.sum(1).tolist())
    def test_max(self):
        r = x.max(1)
        r.reshape((2, 2)) #this is needed cause numpy returns a (2, 2) matrice while lalagrad (2, 1, 2) so we reshape it to be (2, 2)
        self.assertEqual(r.tolist(), a.max(1).tolist())
    def test_min(self):
        r = x.min(1)
        r.reshape((2, 2)) #this is needed cause numpy returns a (2, 2) matrice while lalagrad (2, 1, 2) so we reshape it to be (2, 2)
        self.assertEqual(r.tolist(), a.min(1).tolist())

        
class TestMatrixOps(unittest.TestCase):
    def test_transpose(self): self.assertEqual(m.transpose().tolist(), c.transpose().tolist())
    def test_matmul(self): self.assertEqual(m.matmul(m).tolist(), c.__matmul__(c).tolist())
 
if __name__ == "__main__":
    unittest.main()