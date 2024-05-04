import unittest, math
from lalagrad.tensor import Tensor 


class TestTensorOps(unittest.TestCase):
    def setUp(self): 
       self.t1 = Tensor([[1004.0, 1004.0]])
       self.t2 = Tensor([[22, 1]])
       
    def test_add(self): self.assertEqual((self.t1 + self.t2).data, [x+y for x, y in zip(self.t1.data, self.t2.data)])
    def test_sub(self): self.assertEqual((self.t1 - self.t2).data, [x-y for x, y in zip(self.t1.data, self.t2.data)])
    def test_mul(self): self.assertEqual((self.t1 * self.t2).data, [x*y for x, y in zip(self.t1.data, self.t2.data)])
    def test_log(self): self.assertEqual((self.t1.log(10)).data, [round(math.log10(x), self.t1.dtype.precision) for x in self.t1.data])
    
        
    
if __name__ == "__main__":
    unittest.main()